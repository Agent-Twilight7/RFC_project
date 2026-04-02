"""
Step 15: Temporal consistency result visualization and diagnostics.

Pipeline:
1. Load the finalized step 14 temporal evaluation output.
2. Keep only multi-node paths for summary plots.
3. Plot label counts, stage-wise label distribution, and temporal score histogram.
4. Analyze which sensors most often have null change times.
5. Compare only multi-node paths with complete timestamps.
6. Save plots to reports/figures and analysis tables to data/processed/step15.
"""

import argparse
import json
import os
from collections import Counter
from datetime import datetime
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

STEP14_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "step14")
STEP15_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "step15")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")
RESULTS_MD_PATH = os.path.join(PROJECT_ROOT, "RESULTS.md")

os.makedirs(STEP15_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

STEP14_SUMMARY_PATH = os.path.join(STEP14_DIR, "propagation_temporal_summary.json")
DEFAULT_OUTPUT_CSV = os.path.join(STEP15_DIR, "step15_complete_multinode_paths.csv")
DEFAULT_NULL_SENSOR_CSV = os.path.join(STEP15_DIR, "step15_null_change_time_sensor_counts.csv")
DEFAULT_SUMMARY_JSON = os.path.join(STEP15_DIR, "step15_temporal_analysis_summary.json")


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
        print(f"[STEP 15] Output file is locked. Saved CSV to fallback path: {fallback_path}")
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
        print(f"[STEP 15] Summary file is locked. Saved JSON to fallback path: {fallback_path}")
        return fallback_path


def load_step14_output_path(summary_path: str) -> str:
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Missing step 14 summary file: {summary_path}")

    with open(summary_path, "r") as f:
        summary = json.load(f)

    output_path = summary.get("output_path")
    if not output_path:
        raise ValueError("Step 14 summary JSON does not contain 'output_path'.")
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Step 14 output CSV not found: {output_path}")
    return output_path


def parse_change_times(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    if isinstance(value, float) and pd.isna(value):
        return {}
    if not isinstance(value, str):
        return {}

    text = value.strip()
    if not text:
        return {}

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}

    return parsed if isinstance(parsed, dict) else {}


def add_timestamp_completeness_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["change_times_dict"] = df["change_times"].apply(parse_change_times)
    df["has_complete_timestamps"] = df["change_times_dict"].apply(
        lambda payload: bool(payload) and all(timestamp is not None for timestamp in payload.values())
    )
    return df


def plot_label_counts(df_multinode: pd.DataFrame) -> str:
    save_path = os.path.join(FIGURES_DIR, "step15_temporal_label_counts.png")
    plt.figure(figsize=(8, 5))
    df_multinode["temporal_label"].value_counts().plot(kind="bar", color="steelblue")
    plt.title("Temporal Consistency Labels (Multi-node Paths)")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


def plot_stage_label_distribution(df_multinode: pd.DataFrame) -> str:
    save_path = os.path.join(FIGURES_DIR, "step15_stage_temporal_label_distribution.png")
    stage_label_counts = pd.crosstab(df_multinode["stage"], df_multinode["temporal_label"])
    plt.figure(figsize=(10, 6))
    stage_label_counts.plot(kind="bar", stacked=True, ax=plt.gca(), colormap="tab20c")
    plt.title("Stage-wise Temporal Label Distribution (Multi-node Paths)")
    plt.xlabel("Stage")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


def plot_temporal_score_histogram(df_multinode: pd.DataFrame) -> str:
    save_path = os.path.join(FIGURES_DIR, "step15_temporal_score_histogram.png")
    plt.figure(figsize=(8, 5))
    plt.hist(df_multinode["temporal_score"], bins=20, color="darkorange", edgecolor="black")
    plt.title("Temporal Score Histogram (Multi-node Paths)")
    plt.xlabel("Temporal Score")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


def plot_interface_label_counts(df_multinode: pd.DataFrame) -> str:
    save_path = os.path.join(FIGURES_DIR, "step15_interface_label_counts.png")
    plt.figure(figsize=(8, 5))
    df_multinode["interface_label"].value_counts().plot(kind="bar", color="seagreen")
    plt.title("Interface Sensor Labels (Multi-node Paths)")
    plt.xlabel("Interface Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


def plot_anomaly_within_span_distribution(df_multinode: pd.DataFrame) -> str:
    """Plot distribution of anomaly_within_path_span (True/False) for multi-node paths."""
    save_path = os.path.join(FIGURES_DIR, "step15_anomaly_within_path_span_counts.png")
    plt.figure(figsize=(6, 4))
    (
        df_multinode["anomaly_within_path_span"]
        .astype(str)
        .value_counts()
        .reindex(["True", "False"], fill_value=0)
        .plot(kind="bar", color="mediumpurple")
    )
    plt.title("Anomaly Within Path Span (Multi-node Paths)")
    plt.xlabel("anomaly_within_path_span")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


def _compute_anomaly_position(row: pd.Series) -> str:
    """Classify anomaly position relative to path span as before/within/after/other."""
    before = bool(row.get("anomaly_before_path"))
    after = bool(row.get("anomaly_after_path"))
    within = bool(row.get("anomaly_within_path_span"))

    if before:
        return "before"
    if after:
        return "after"
    if within:
        return "within"
    return "other"


def plot_anomaly_position_counts(df_multinode: pd.DataFrame) -> str:
    """Plot counts of anomaly position categories (before/within/after/other)."""
    save_path = os.path.join(FIGURES_DIR, "step15_anomaly_position_counts.png")
    plt.figure(figsize=(6, 4))
    df_multinode["anomaly_position"].value_counts().plot(kind="bar", color="slateblue")
    plt.title("Anomaly vs Path Span Position (Multi-node Paths)")
    plt.xlabel("Position")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


def analyze_null_change_time_sensors(df_multinode: pd.DataFrame) -> pd.DataFrame:
    null_counter: Counter[str] = Counter()

    for payload in df_multinode["change_times_dict"]:
        for sensor_name, timestamp in payload.items():
            if timestamp is None:
                null_counter[str(sensor_name)] += 1

    null_df = pd.DataFrame(
        [{"sensor_name": sensor_name, "null_change_time_count": count} for sensor_name, count in null_counter.items()]
    )
    if not null_df.empty:
        null_df = null_df.sort_values("null_change_time_count", ascending=False)
    return null_df


def build_analysis_summary(df_multinode: pd.DataFrame, df_complete: pd.DataFrame, step14_csv_path: str, figure_paths: Dict[str, str]) -> Dict[str, Any]:
    multinode_label_counts = df_multinode["temporal_label"].value_counts().to_dict()
    complete_label_counts = df_complete["temporal_label"].value_counts().to_dict() if not df_complete.empty else {}
    multinode_time_order_counts = df_multinode["time_order_label"].value_counts().to_dict() if "time_order_label" in df_multinode.columns else {}
    complete_time_order_counts = df_complete["time_order_label"].value_counts().to_dict() if "time_order_label" in df_complete.columns else {}
    anomaly_aligned_fraction = float(df_multinode["anomaly_aligned"].mean()) if "anomaly_aligned" in df_multinode.columns and not df_multinode.empty else 0.0
    complete_anomaly_aligned_fraction = float(df_complete["anomaly_aligned"].mean()) if "anomaly_aligned" in df_complete.columns and not df_complete.empty else 0.0

    if "anomaly_within_path_span" in df_multinode.columns:
        anomaly_within_counts = df_multinode["anomaly_within_path_span"].value_counts().to_dict()
    else:
        anomaly_within_counts = {}

    if "anomaly_position" in df_multinode.columns:
        anomaly_position_counts = df_multinode["anomaly_position"].value_counts().to_dict()
    else:
        anomaly_position_counts = {}

    return {
        "step14_output_path": step14_csv_path,
        "total_multinode_paths": int(len(df_multinode)),
        "multinode_label_counts": multinode_label_counts,
        "multinode_time_order_counts": multinode_time_order_counts,
        "multinode_mean_temporal_score": float(df_multinode["temporal_score"].mean()) if not df_multinode.empty else 0.0,
        "multinode_anomaly_aligned_fraction": anomaly_aligned_fraction,
        "multinode_anomaly_within_path_span_counts": anomaly_within_counts,
        "multinode_anomaly_position_counts": anomaly_position_counts,
        "complete_multinode_paths": int(len(df_complete)),
        "complete_multinode_fraction": float(len(df_complete) / len(df_multinode)) if len(df_multinode) > 0 else 0.0,
        "complete_multinode_label_counts": complete_label_counts,
        "complete_multinode_time_order_counts": complete_time_order_counts,
        "complete_multinode_mean_temporal_score": float(df_complete["temporal_score"].mean()) if not df_complete.empty else 0.0,
        "complete_multinode_anomaly_aligned_fraction": complete_anomaly_aligned_fraction,
        "figure_paths": figure_paths,
    }


def _format_label_counts_markdown(label_counts: Dict[str, int]) -> str:
    """Format temporal label counts as a small Markdown table.

    The order is fixed to keep the RESULTS.md section stable across runs.
    """
    labels_in_order = [
        "consistent",
        "time_order_supported",
        "inconsistent",
        "insufficient_evidence",
    ]
    lines = ["| Label | Count |", "|---|---:|"]
    for label in labels_in_order:
        count = int(label_counts.get(label, 0))
        lines.append(f"| `{label}` | {count} |")
    return "\n".join(lines)


def update_results_markdown_with_label_counts(label_counts: Dict[str, int]) -> None:
    """Update the temporal label counts table in RESULTS.md.

    This function replaces the content between the STEP15_TEMPORAL_LABEL_COUNTS
    markers with a freshly generated Markdown table based on the current run.
    If RESULTS.md or the markers are missing, it logs a message and returns.
    """
    if not os.path.exists(RESULTS_MD_PATH):
        print(f"[STEP 15] RESULTS.md not found at {RESULTS_MD_PATH}; skipping docs update.")
        return

    with open(RESULTS_MD_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    start_marker = "<!-- STEP15_TEMPORAL_LABEL_COUNTS_START -->"
    end_marker = "<!-- STEP15_TEMPORAL_LABEL_COUNTS_END -->"

    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)

    if start_idx == -1 or end_idx == -1:
        print("[STEP 15] Temporal label count markers not found in RESULTS.md; skipping docs update.")
        return

    end_idx += len(end_marker)

    table_md = _format_label_counts_markdown(label_counts)
    replacement_block = f"{start_marker}\n{table_md}\n{end_marker}"

    new_text = text[:start_idx] + replacement_block + text[end_idx:]

    with open(RESULTS_MD_PATH, "w", encoding="utf-8") as f:
        f.write(new_text)

    print("[STEP 15] Updated temporal label counts table in RESULTS.md.")


def run_step15(summary_path: str = STEP14_SUMMARY_PATH) -> None:
    step14_csv_path = load_step14_output_path(summary_path)
    print(f"[STEP 15] Loading step 14 output: {step14_csv_path}")

    df = pd.read_csv(step14_csv_path)
    df = add_timestamp_completeness_columns(df)

    df_multinode = df[df["path_length"] >= 2].copy()
    df_complete = df_multinode[df_multinode["has_complete_timestamps"]].copy()

    # If span diagnostics are present, derive summary categories for anomaly position.
    anomaly_within_counts = {}
    anomaly_position_counts = {}
    anomaly_within_figure = None
    anomaly_position_figure = None

    if "anomaly_within_path_span" in df_multinode.columns:
        anomaly_within_counts = df_multinode["anomaly_within_path_span"].value_counts().to_dict()
        print("[STEP 15] anomaly_within_path_span counts (multi-node paths):")
        for key, value in sorted(anomaly_within_counts.items()):
            print(f"  {key}: {value}")

        anomaly_within_figure = plot_anomaly_within_span_distribution(df_multinode)
        print(f"[STEP 15] Saved anomaly_within_path_span plot to: {anomaly_within_figure}")

    if {"anomaly_before_path", "anomaly_after_path", "anomaly_within_path_span"}.issubset(df_multinode.columns):
        df_multinode = df_multinode.copy()
        df_multinode["anomaly_position"] = df_multinode.apply(_compute_anomaly_position, axis=1)
        anomaly_position_counts = df_multinode["anomaly_position"].value_counts().to_dict()
        print("[STEP 15] Anomaly position counts (before/within/after/other):")
        for key, value in sorted(anomaly_position_counts.items()):
            print(f"  {key}: {value}")

        anomaly_position_figure = plot_anomaly_position_counts(df_multinode)
        print(f"[STEP 15] Saved anomaly position plot to: {anomaly_position_figure}")

    # Print label counts to the console for quick inspection.
    multinode_label_counts = df_multinode["temporal_label"].value_counts().to_dict()
    print("[STEP 15] Temporal label counts for multi-node paths:")
    for label, count in sorted(multinode_label_counts.items()):
        print(f"  {label}: {count}")

    label_count_figure = plot_label_counts(df_multinode)
    print(f"[STEP 15] Saved label counts plot to: {label_count_figure}")

    stage_label_figure = plot_stage_label_distribution(df_multinode)
    print(f"[STEP 15] Saved stage-wise label plot to: {stage_label_figure}")

    score_histogram_figure = plot_temporal_score_histogram(df_multinode)
    print(f"[STEP 15] Saved temporal score histogram to: {score_histogram_figure}")

    interface_label_figure = plot_interface_label_counts(df_multinode)
    print(f"[STEP 15] Saved interface label counts plot to: {interface_label_figure}")

    null_sensor_df = analyze_null_change_time_sensors(df_multinode)
    null_sensor_csv_path = write_csv_with_fallback(null_sensor_df, DEFAULT_NULL_SENSOR_CSV)
    print(f"[STEP 15] Saved null change-time sensor counts to: {null_sensor_csv_path}")

    complete_output_csv_path = write_csv_with_fallback(df_complete, DEFAULT_OUTPUT_CSV)
    print(f"[STEP 15] Saved complete multi-node path comparison table to: {complete_output_csv_path}")

    summary = build_analysis_summary(
        df_multinode=df_multinode,
        df_complete=df_complete,
        step14_csv_path=step14_csv_path,
        figure_paths={
            "label_counts": label_count_figure,
            "stage_label_distribution": stage_label_figure,
            "temporal_score_histogram": score_histogram_figure,
            "interface_label_counts": interface_label_figure,
            "anomaly_within_path_span_counts": anomaly_within_figure,
            "anomaly_position_counts": anomaly_position_figure,
        },
    )
    summary["null_sensor_csv_path"] = null_sensor_csv_path
    summary["complete_multinode_csv_path"] = complete_output_csv_path
    summary_json_path = write_json_with_fallback(summary, DEFAULT_SUMMARY_JSON)
    print(f"[STEP 15] Saved analysis summary to: {summary_json_path}")

    # Finally, refresh the temporal label counts table in RESULTS.md.
    update_results_markdown_with_label_counts(multinode_label_counts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize and analyze step 14 temporal consistency results.")
    parser.add_argument("--step14-summary", default=STEP14_SUMMARY_PATH, help="Path to step 14 summary JSON.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_step15(summary_path=args.step14_summary)