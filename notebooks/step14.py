"""
Step 14: Temporal consistency evaluation for propagation paths.

Pipeline:
1. Load candidate propagation paths from step 12.
2. Load window timestamps from step 5.
3. Load per-feature anomaly scores from step 7.
4. Recompute feature thresholds using normal windows, consistent with step 9.
5. Derive change/onset times for each sensor along each candidate path.
6. Evaluate whether each path is temporally consistent with the anomaly time.
7. Save results under data/processed/step14 without modifying earlier outputs.
"""

import argparse
import ast
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

STEP5_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "step5")
STEP6_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "step6")
STEP7_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "step7")
STEP12_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "step12")
STEP14_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "step14")
os.makedirs(STEP14_DIR, exist_ok=True)

DEFAULT_INPUT_PATH = os.path.join(STEP12_DIR, "swat_rca_step12_results.csv")
DEFAULT_OUTPUT_PATH = os.path.join(STEP14_DIR, "propagation_path_temporal_evaluation.csv")
DEFAULT_SUMMARY_PATH = os.path.join(STEP14_DIR, "propagation_temporal_summary.json")

LOOKBACK_SECONDS = int(os.getenv("STEP14_LOOKBACK_SECONDS", "300"))
EPSILON_SECONDS = int(os.getenv("STEP14_EPSILON_SECONDS", "5"))
ANOMALY_WINDOW_SECONDS = int(os.getenv("STEP14_ANOMALY_WINDOW_SECONDS", "60"))
STEP14_PROGRESS_EVERY = int(os.getenv("STEP14_PROGRESS_EVERY", "100"))

RawTime = Union[int, float, str, pd.Timestamp, datetime]
NumericTime = Union[int, float]


def _load_structured_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, (list, dict, tuple)):
        return value
    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text:
        return None

    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except (json.JSONDecodeError, ValueError, SyntaxError):
            continue

    return text


def parse_candidate_paths(value: Any) -> List[List[str]]:
    parsed = _load_structured_value(value)
    if parsed is None:
        return []

    if isinstance(parsed, str):
        nodes = [part.strip() for part in parsed.split("->") if part.strip()]
        return [nodes] if nodes else []

    if isinstance(parsed, (list, tuple)):
        if not parsed:
            return []

        first = parsed[0]
        if isinstance(first, (list, tuple)):
            return [
                [str(node).strip() for node in path if str(node).strip()]
                for path in parsed
                if path
            ]

        return [[str(node).strip() for node in parsed if str(node).strip()]]

    return [[str(parsed).strip()]]


def parse_step12_paths(df: pd.DataFrame) -> pd.DataFrame:
    expanded_rows: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        row_dict = row.to_dict()
        candidate_paths = parse_candidate_paths(row_dict.get("propagation_paths"))

        for path_index, path_nodes in enumerate(candidate_paths, start=1):
            expanded_row = {
                "anomaly_id": row_dict.get("anomaly_id"),
                "stage": row_dict.get("stage"),
                "timestamp": row_dict.get("timestamp"),
                "phase": row_dict.get("phase"),
                "guilty_feature": row_dict.get("guilty_feature"),
                "candidate_path_index": path_index,
                "candidate_path_nodes": path_nodes,
                "candidate_path": " -> ".join(path_nodes),
            }
            expanded_rows.append(expanded_row)

    return pd.DataFrame(expanded_rows)


def _coerce_time(value: object) -> Optional[Union[NumericTime, pd.Timestamp]]:
    """Convert a supported timestamp value to either numeric or pandas Timestamp."""
    if value is None:
        return None

    if isinstance(value, pd.Timestamp):
        return value

    if isinstance(value, datetime):
        return pd.Timestamp(value)

    if isinstance(value, (int, float)):
        return value

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            timestamp = pd.to_datetime(text, errors="coerce")
            if pd.isna(timestamp):
                return None
            return timestamp

    return None


def load_sensor_stages() -> Dict[str, str]:
    """Load a global sensor->stage mapping from the stage_feature_map JSON.

    The JSON is produced in step 6 and has the shape {"P1": [sensors...], ...}.
    """
    stage_map_path = os.path.join(STEP6_DIR, "stage_feature_map.json")
    if not os.path.exists(stage_map_path):
        raise FileNotFoundError(f"Missing stage_feature_map.json in step6 directory: {stage_map_path}")

    with open(stage_map_path, "r") as f:
        stage_map = json.load(f)

    sensor_stages: Dict[str, str] = {}
    for stage, sensors in stage_map.items():
        for sensor_name in sensors:
            sensor_stages[str(sensor_name)] = str(stage)

    return sensor_stages


def validate_interface_sensor(
    path: List[str],
    sensor_stages: Dict[str, str],
    change_times: Dict[str, Optional[RawTime]],
    epsilon: NumericTime = 5,
) -> Tuple[str, str]:
    """Validate temporal ordering at stage-boundary (interface) sensors.

    Returns (label, reason) where label is one of:
    - "consistent": all interface sensors respect temporal order within tolerance
    - "inconsistent": at least one interface sensor violates the order
    - "insufficient_evidence": missing stages/timestamps, no interfaces, or mixed formats
    """
    if path is None or len(path) < 3:
        return "insufficient_evidence", "path must contain at least three sensors"

    # Ensure all sensors have a stage and a timestamp
    coerced_times: Dict[str, Optional[Union[NumericTime, pd.Timestamp]]] = {}
    for sensor_name in path:
        if sensor_name not in sensor_stages:
            return "insufficient_evidence", f"missing stage for sensor {sensor_name}"

        raw_time = change_times.get(sensor_name)
        sensor_time = _coerce_time(raw_time)
        if sensor_time is None:
            return "insufficient_evidence", f"missing or invalid timestamp for sensor {sensor_name}"
        coerced_times[sensor_name] = sensor_time

    # Determine time representation and guard against mixed formats
    any_ts = any(isinstance(t, pd.Timestamp) for t in coerced_times.values())
    any_num = any(isinstance(t, (int, float)) for t in coerced_times.values())

    if any_ts and any_num:
        return "insufficient_evidence", "mixed timestamp formats across sensors"

    use_datetime = any_ts
    if use_datetime:
        epsilon_delta = pd.Timedelta(seconds=float(epsilon))
    else:
        eps = float(epsilon)

    # Identify interface sensors: stage different from both neighbors
    interface_indices: List[int] = []
    for i in range(1, len(path) - 1):
        prev_sensor = path[i - 1]
        sensor = path[i]
        next_sensor = path[i + 1]
        stage_prev = sensor_stages.get(prev_sensor)
        stage_curr = sensor_stages.get(sensor)
        stage_next = sensor_stages.get(next_sensor)

        if stage_curr is None or stage_prev is None or stage_next is None:
            return "insufficient_evidence", "missing stage metadata for interface triple"

        if stage_curr != stage_prev and stage_curr != stage_next:
            interface_indices.append(i)

    if not interface_indices:
        return "insufficient_evidence", "no interface sensors in path"

    # Check temporal order around each interface sensor
    for i in interface_indices:
        prev_sensor = path[i - 1]
        sensor = path[i]
        next_sensor = path[i + 1]

        t_prev = coerced_times.get(prev_sensor)
        t_curr = coerced_times.get(sensor)
        t_next = coerced_times.get(next_sensor)

        if t_prev is None or t_curr is None or t_next is None:
            return "insufficient_evidence", "missing timestamp within interface triple"

        if use_datetime:
            if not (t_curr >= t_prev - epsilon_delta and t_next >= t_curr - epsilon_delta):
                reason = f"interface sensor {sensor} violates temporal order between {prev_sensor} and {next_sensor}"
                return "inconsistent", reason
        else:
            if not (t_curr >= t_prev - eps and t_next >= t_curr - eps):
                reason = f"interface sensor {sensor} violates temporal order between {prev_sensor} and {next_sensor}"
                return "inconsistent", reason

    return "consistent", "all interface sensors follow temporal order within tolerance"


def evaluate_propagation_path(
    path: List[str],
    change_times: Dict[str, Optional[RawTime]],
    anomaly_time: RawTime,
    epsilon: NumericTime = 5,
    anomaly_window: NumericTime = 50,
) -> Dict[str, Any]:
    """
    Evaluate the temporal consistency of an ordered propagation path.

    The function supports both numeric times and datetime-like timestamps.
    If datetime values are used, epsilon and anomaly_window are interpreted in seconds.

    Returns a structured result separating:
    1. time-order support across edges
    2. last-node alignment with the anomaly time
    3. an overall interpretation label
    """
    if path is None or len(path) < 2:
        return {
            "temporal_label": "insufficient_evidence",
            "temporal_score": 0.0,
            "temporal_reason": "path must contain at least two sensors",
            "ordered_edges": 0,
            "total_edges": 0,
            "anomaly_aligned": False,
            "alignment_gap_seconds": None,
            "time_order_label": "insufficient_evidence",
        }

    anomaly_time_value = _coerce_time(anomaly_time)
    if anomaly_time_value is None:
        return {
            "temporal_label": "insufficient_evidence",
            "temporal_score": 0.0,
            "temporal_reason": "invalid anomaly time",
            "ordered_edges": 0,
            "total_edges": len(path) - 1,
            "anomaly_aligned": False,
            "alignment_gap_seconds": None,
            "time_order_label": "insufficient_evidence",
        }

    ordered_times: List[Union[NumericTime, pd.Timestamp]] = []
    for sensor_name in path:
        if sensor_name not in change_times:
            return {
                "temporal_label": "insufficient_evidence",
                "temporal_score": 0.0,
                "temporal_reason": "missing timestamp",
                "ordered_edges": 0,
                "total_edges": len(path) - 1,
                "anomaly_aligned": False,
                "alignment_gap_seconds": None,
                "time_order_label": "insufficient_evidence",
            }

        sensor_time = _coerce_time(change_times.get(sensor_name))
        if sensor_time is None:
            return {
                "temporal_label": "insufficient_evidence",
                "temporal_score": 0.0,
                "temporal_reason": "missing timestamp",
                "ordered_edges": 0,
                "total_edges": len(path) - 1,
                "anomaly_aligned": False,
                "alignment_gap_seconds": None,
                "time_order_label": "insufficient_evidence",
            }

        ordered_times.append(sensor_time)

    use_datetime = isinstance(anomaly_time_value, pd.Timestamp)
    if use_datetime and any(not isinstance(item, pd.Timestamp) for item in ordered_times):
        return {
            "temporal_label": "insufficient_evidence",
            "temporal_score": 0.0,
            "temporal_reason": "mixed timestamp formats",
            "ordered_edges": 0,
            "total_edges": len(path) - 1,
            "anomaly_aligned": False,
            "alignment_gap_seconds": None,
            "time_order_label": "insufficient_evidence",
        }
    if not use_datetime and any(isinstance(item, pd.Timestamp) for item in ordered_times):
        return {
            "temporal_label": "insufficient_evidence",
            "temporal_score": 0.0,
            "temporal_reason": "mixed timestamp formats",
            "ordered_edges": 0,
            "total_edges": len(path) - 1,
            "anomaly_aligned": False,
            "alignment_gap_seconds": None,
            "time_order_label": "insufficient_evidence",
        }

    total_edges = len(path) - 1
    ordered_edges = 0

    # Compute path time span for diagnostics
    path_start_time = min(ordered_times)
    path_end_time = max(ordered_times)

    if use_datetime:
        epsilon_delta = pd.Timedelta(seconds=float(epsilon))
        anomaly_window_delta = pd.Timedelta(seconds=float(anomaly_window))

        for current_time, next_time in zip(ordered_times, ordered_times[1:]):
            if next_time >= current_time - epsilon_delta:
                ordered_edges += 1

        last_time = ordered_times[-1]
        anomaly_aligned = abs(last_time - anomaly_time_value) <= anomaly_window_delta
        alignment_gap_seconds = float(abs((last_time - anomaly_time_value).total_seconds()))

        # Diagnostics: relationship between anomaly time and path span
        anomaly_before_path = anomaly_time_value < (path_start_time - epsilon_delta)
        anomaly_after_path = anomaly_time_value > (path_end_time + epsilon_delta)
    else:
        for current_time, next_time in zip(ordered_times, ordered_times[1:]):
            if next_time >= current_time - epsilon:
                ordered_edges += 1

        last_time = ordered_times[-1]
        anomaly_aligned = abs(last_time - anomaly_time_value) <= anomaly_window
        alignment_gap_seconds = float(abs(last_time - anomaly_time_value))

        # Diagnostics: relationship between anomaly time and path span
        eps = float(epsilon)
        anomaly_before_path = anomaly_time_value < (path_start_time - eps)
        anomaly_after_path = anomaly_time_value > (path_end_time + eps)

    temporal_score = ordered_edges / total_edges

    # Anomaly-position diagnostic relative to the path span
    anomaly_within_path_span = not anomaly_before_path and not anomaly_after_path

    if temporal_score >= 0.8:
        time_order_label = "strong"
    elif temporal_score <= 0.3:
        time_order_label = "weak"
    else:
        time_order_label = "mixed"

    if temporal_score <= 0.3:
        temporal_label = "inconsistent"
        temporal_reason = f"only {ordered_edges}/{total_edges} edges follow the path order"
    elif temporal_score >= 0.8 and anomaly_aligned:
        temporal_label = "consistent"
        temporal_reason = (
            f"{ordered_edges}/{total_edges} edges follow the path order and the last node is near the anomaly time"
        )
    elif temporal_score >= 0.8 and not anomaly_aligned:
        temporal_label = "time_order_supported"
        temporal_reason = (
            f"{ordered_edges}/{total_edges} edges follow the path order, but the last node is not near the anomaly time"
        )
    else:
        temporal_label = "insufficient_evidence"
        temporal_reason = f"time order is partial with {ordered_edges}/{total_edges} ordered edges"

    return {
        "temporal_label": temporal_label,
        "temporal_score": temporal_score,
        "temporal_reason": temporal_reason,
        "ordered_edges": ordered_edges,
        "total_edges": total_edges,
        "anomaly_aligned": bool(anomaly_aligned),
        "alignment_gap_seconds": alignment_gap_seconds,
        "time_order_label": time_order_label,
        "path_start_time": path_start_time,
        "path_end_time": path_end_time,
        "anomaly_before_path": bool(anomaly_before_path),
        "anomaly_after_path": bool(anomaly_after_path),
        "anomaly_within_path_span": bool(anomaly_within_path_span),
    }


def load_window_metadata() -> pd.DataFrame:
    window_end_times = np.load(os.path.join(STEP5_DIR, "window_end_times.npy"), allow_pickle=True)
    window_phase = np.load(os.path.join(STEP5_DIR, "window_phase.npy"), allow_pickle=True)

    df_meta = pd.DataFrame({
        "t_stamp": pd.to_datetime(window_end_times),
        "phase": window_phase,
    })
    return df_meta


def load_stage_feature_scores(stage: str) -> pd.DataFrame:
    score_path = os.path.join(STEP7_DIR, f"swat_{stage}_anomaly_scores.csv")
    if not os.path.exists(score_path):
        raise FileNotFoundError(f"Missing feature score file for {stage}: {score_path}")
    return pd.read_csv(score_path)


def compute_feature_thresholds(df_scores: pd.DataFrame, normal_mask: pd.Series) -> pd.Series:
    normal_indices = np.where(normal_mask.to_numpy())[0]
    if len(normal_indices) > 0:
        normal_df = df_scores.iloc[normal_indices]
    else:
        normal_df = df_scores
    return normal_df.quantile(0.999)


def derive_sensor_change_time(
    timestamps: pd.Series,
    sensor_scores: pd.Series,
    threshold: float,
    anomaly_time: pd.Timestamp,
    lookback_seconds: int,
) -> Optional[pd.Timestamp]:
    lookback_start = anomaly_time - pd.Timedelta(seconds=lookback_seconds)
    mask = (timestamps <= anomaly_time) & (timestamps >= lookback_start)

    if not mask.any():
        return None

    score_window = sensor_scores.loc[mask].reset_index(drop=True)
    time_window = timestamps.loc[mask].reset_index(drop=True)
    above_threshold = (score_window > threshold).to_numpy()

    if not above_threshold.any():
        return None

    transition_mask = above_threshold & np.concatenate(([True], ~above_threshold[:-1]))
    transition_indices = np.where(transition_mask)[0]

    if len(transition_indices) > 0:
        onset_index = int(transition_indices[0]) # Take the first transition from below to above threshold - CHECK IF THIS IS CORRECT LOGIC
    else:
        onset_index = int(np.where(above_threshold)[0][0])

    return pd.Timestamp(time_window.iloc[onset_index])


def build_change_times_for_path(
    stage_scores: pd.DataFrame,
    feature_thresholds: pd.Series,
    timestamps: pd.Series,
    path_nodes: List[str],
    anomaly_time: pd.Timestamp,
    lookback_seconds: int,
) -> Dict[str, Optional[pd.Timestamp]]:
    change_times: Dict[str, Optional[pd.Timestamp]] = {}

    for sensor_name in path_nodes:
        if sensor_name not in stage_scores.columns:
            change_times[sensor_name] = None
            continue

        threshold = float(feature_thresholds.get(sensor_name, np.inf))
        sensor_scores = stage_scores[sensor_name]
        change_times[sensor_name] = derive_sensor_change_time(
            timestamps=timestamps,
            sensor_scores=sensor_scores,
            threshold=threshold,
            anomaly_time=anomaly_time,
            lookback_seconds=lookback_seconds,
        )

    return change_times


def serialize_change_times(change_times: Dict[str, Optional[pd.Timestamp]]) -> str:
    payload = {
        sensor_name: None if timestamp is None else timestamp.isoformat()
        for sensor_name, timestamp in change_times.items()
    }
    return json.dumps(payload)


def build_fallback_output_path(path: str) -> str:
    base, ext = os.path.splitext(path)
    timestamp_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base}_{timestamp_suffix}{ext}"


def write_csv_with_fallback(df: pd.DataFrame, path: str) -> str:
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        fallback_path = build_fallback_output_path(path)
        df.to_csv(fallback_path, index=False)
        print(f"[STEP 14] Output file is locked. Saved CSV to fallback path: {fallback_path}")
        return fallback_path


def write_json_with_fallback(payload: Dict[str, Any], path: str) -> str:
    try:
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        return path
    except PermissionError:
        fallback_path = build_fallback_output_path(path)
        with open(fallback_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[STEP 14] Summary file is locked. Saved JSON to fallback path: {fallback_path}")
        return fallback_path


def evaluate_step12_paths(
    input_path: str = DEFAULT_INPUT_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    summary_path: str = DEFAULT_SUMMARY_PATH,
    lookback_seconds: int = LOOKBACK_SECONDS,
    epsilon_seconds: int = EPSILON_SECONDS,
    anomaly_window_seconds: int = ANOMALY_WINDOW_SECONDS,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    print(f"[STEP 14] Loading step 12 propagation paths from: {input_path}")
    df_step12 = pd.read_csv(input_path)
    df_paths = parse_step12_paths(df_step12)

    if df_paths.empty:
        empty_df = pd.DataFrame(
            columns=[
                "anomaly_id",
                "stage",
                "timestamp",
                "phase",
                "candidate_path_index",
                "candidate_path",
                "change_times",
                "temporal_label",
                "temporal_score",
                "temporal_reason",
            ]
        )
        actual_output_path = write_csv_with_fallback(empty_df, output_path)
        actual_summary_path = write_json_with_fallback({"total_paths": 0, "output_path": actual_output_path}, summary_path)
        print(f"[STEP 14] Saved empty temporal evaluation CSV to: {actual_output_path}")
        print(f"[STEP 14] Saved empty summary JSON to: {actual_summary_path}")
        return empty_df

    if max_rows is not None:
        df_paths = df_paths.head(max_rows).copy()

    df_meta = load_window_metadata()
    timestamps = df_meta["t_stamp"]
    normal_mask = df_meta["phase"] == "normal"

    sensor_stages = load_sensor_stages()

    stage_cache: Dict[str, Dict[str, Any]] = {}
    results: List[Dict[str, Any]] = []
    total_rows = len(df_paths)

    for index, row in enumerate(df_paths.to_dict(orient="records"), start=1):
        stage = str(row.get("stage"))
        if stage not in stage_cache:
            stage_scores = load_stage_feature_scores(stage)
            feature_thresholds = compute_feature_thresholds(stage_scores, normal_mask)
            stage_cache[stage] = {
                "scores": stage_scores,
                "thresholds": feature_thresholds,
            }

        anomaly_time = pd.to_datetime(row.get("timestamp"), errors="coerce")
        path_nodes = row.get("candidate_path_nodes", [])
        stage_scores = stage_cache[stage]["scores"]
        feature_thresholds = stage_cache[stage]["thresholds"]

        change_times = build_change_times_for_path(
            stage_scores=stage_scores,
            feature_thresholds=feature_thresholds,
            timestamps=timestamps,
            path_nodes=path_nodes,
            anomaly_time=anomaly_time,
            lookback_seconds=lookback_seconds,
        )

        evaluation = evaluate_propagation_path(
            path=path_nodes,
            change_times=change_times,
            anomaly_time=anomaly_time,
            epsilon=epsilon_seconds,
            anomaly_window=anomaly_window_seconds,
        )

        interface_label, interface_reason = validate_interface_sensor(
            path=path_nodes,
            sensor_stages=sensor_stages,
            change_times=change_times,
            epsilon=epsilon_seconds,
        )

        results.append(
            {
                "anomaly_id": row.get("anomaly_id"),
                "stage": stage,
                "timestamp": row.get("timestamp"),
                "phase": row.get("phase"),
                "guilty_feature": row.get("guilty_feature"),
                "candidate_path_index": row.get("candidate_path_index"),
                "candidate_path": row.get("candidate_path"),
                "path_length": len(path_nodes),
                "change_times": serialize_change_times(change_times),
                "temporal_label": evaluation["temporal_label"],
                "temporal_score": evaluation["temporal_score"],
                "temporal_reason": evaluation["temporal_reason"],
                "ordered_edges": evaluation["ordered_edges"],
                "total_edges": evaluation["total_edges"],
                "anomaly_aligned": evaluation["anomaly_aligned"],
                "alignment_gap_seconds": evaluation["alignment_gap_seconds"],
                "time_order_label": evaluation["time_order_label"],
                "path_start_time": evaluation.get("path_start_time"),
                "path_end_time": evaluation.get("path_end_time"),
                "anomaly_before_path": evaluation.get("anomaly_before_path"),
                "anomaly_after_path": evaluation.get("anomaly_after_path"),
                "anomaly_within_path_span": evaluation.get("anomaly_within_path_span"),
                "interface_label": interface_label,
                "interface_reason": interface_reason,
            }
        )

        if index % max(1, STEP14_PROGRESS_EVERY) == 0 or index == total_rows:
            print(f"[STEP 14] Progress: {index}/{total_rows} paths evaluated")

    output_df = pd.DataFrame(results)
    actual_output_path = write_csv_with_fallback(output_df, output_path)

    summary_df = output_df[output_df["path_length"] >= 2].copy()
    excluded_single_node_paths = int((output_df["path_length"] < 2).sum())

    summary = {
        "total_paths": int(len(output_df)),
        "paths_used_in_summary": int(len(summary_df)),
        "excluded_single_node_paths": excluded_single_node_paths,
        "label_counts": summary_df["temporal_label"].value_counts().to_dict(),
        "mean_temporal_score": float(summary_df["temporal_score"].mean()) if not summary_df.empty else 0.0,
        "input_path": input_path,
        "output_path": actual_output_path,
        "lookback_seconds": lookback_seconds,
        "epsilon_seconds": epsilon_seconds,
        "anomaly_window_seconds": anomaly_window_seconds,
    }
    actual_summary_path = write_json_with_fallback(summary, summary_path)

    print(f"[STEP 14] Saved temporal evaluation CSV to: {actual_output_path}")
    print(f"[STEP 14] Saved summary JSON to: {actual_summary_path}")
    return output_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate temporal consistency of step 12 propagation paths.")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="Input step 12 CSV path.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Output CSV path.")
    parser.add_argument("--summary", default=DEFAULT_SUMMARY_PATH, help="Output summary JSON path.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row limit for quick runs.")
    parser.add_argument("--lookback-seconds", type=int, default=LOOKBACK_SECONDS, help="Lookback window for onset detection.")
    parser.add_argument("--epsilon-seconds", type=int, default=EPSILON_SECONDS, help="Tolerance for pairwise time order.")
    parser.add_argument("--anomaly-window-seconds", type=int, default=ANOMALY_WINDOW_SECONDS, help="Allowed distance between last node time and anomaly time.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_step12_paths(
        input_path=args.input,
        output_path=args.output,
        summary_path=args.summary,
        lookback_seconds=args.lookback_seconds,
        epsilon_seconds=args.epsilon_seconds,
        anomaly_window_seconds=args.anomaly_window_seconds,
        max_rows=args.max_rows,
    )