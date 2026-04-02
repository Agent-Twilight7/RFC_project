import argparse
import ast
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

# Read API key from environment for safety.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Project-relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

os.makedirs(os.path.join(PROJECT_ROOT, "data", "processed", "step16"), exist_ok=True)

STEP12_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "step12", "swat_rca_step12_results.csv")
STEP13_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "step13", "propagation_path_llm_evaluation.csv")
STEP14_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "step14", "propagation_path_temporal_evaluation.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "step16", "llm_explanations-api-modified.csv")

STEP16_MAX_RETRIES = 3
STEP16_RETRY_DELAY = 2
STEP16_LLM_WORKERS = 1
STEP16_MAX_LLM_CALLS = 0

WEIGHT_LLM = 0.4
WEIGHT_TEMPORAL = 0.4
WEIGHT_RCA = 0.2

TEMPORAL_LABEL_PRIORITY = {
    "consistent": 4,
    "time_order_supported": 3,
    "insufficient_evidence": 2,
    "inconsistent": 1,
}

NO_EXPLANATION_TEXT = "No reliable causal explanation found"
OUTPUT_COLUMNS = [
    "anomaly_id",
    "stage",
    "timestamp",
    "best_path",
    "path_confidence_score",
    "llm_confidence",
    "temporal_score",
    "rca_score",
    "final_score",
    "summary",
    "root_cause_explanation",
    "propagation_explanation",
    "confidence_explanation",
    "recommendation",
]

SWAT_DATASET_CONTEXT = (
    "SWaT (Secure Water Treatment) is a cyber-physical industrial water-treatment dataset "
    "with six process stages (P1-P6). In this repository, anomaly detection and RCA are based "
    "on continuous process-value sensors ending in .Pv."
)

STAGE_CONTEXT = {
    "P1": {
        "name": "Raw Water Intake",
        "description": "Intake-side behavior focused on water level and inlet flow.",
        "modeled_variables": ["LIT101.Pv", "FIT101.Pv"],
    },
    "P2": {
        "name": "Pre-treatment",
        "description": "Pre-treatment behavior with flow plus analyzer measurements.",
        "modeled_variables": ["FIT201.Pv", "AIT201.Pv", "AIT202.Pv", "AIT203.Pv"],
    },
    "P3": {
        "name": "Ultra-Filtration",
        "description": "Filtration behavior combining analyzer, level, flow, and differential pressure.",
        "modeled_variables": ["AIT301.Pv", "AIT302.Pv", "AIT303.Pv", "LIT301.Pv", "FIT301.Pv", "DPIT301.Pv"],
    },
    "P4": {
        "name": "De-Chlorination",
        "description": "De-chlorination behavior with analyzer, level, and flow measurements.",
        "modeled_variables": ["LIT401.Pv", "FIT401.Pv", "AIT401.Pv", "AIT402.Pv"],
    },
    "P5": {
        "name": "Reverse Osmosis",
        "description": "Membrane-stage behavior with analyzer, flow, and pressure measurements.",
        "modeled_variables": [
            "FIT501.Pv", "FIT502.Pv", "FIT503.Pv", "FIT504.Pv", "AIT501.Pv", "AIT502.Pv",
            "AIT503.Pv", "AIT504.Pv", "PIT501.Pv", "PIT502.Pv", "PIT503.Pv",
        ],
    },
    "P6": {
        "name": "Disposition",
        "description": "Final stage with a small continuous measurement set.",
        "modeled_variables": ["FIT601.Pv"],
    },
}

SENSOR_PREFIX_CONTEXT = {
    "FIT": "Flow Indicator Transmitter (flow-related measurement)",
    "LIT": "Level Indicator Transmitter (level-related measurement)",
    "AIT": "Analyzer Indicator Transmitter (quality/chemistry measurement)",
    "PIT": "Pressure Indicator Transmitter (pressure measurement)",
    "DPIT": "Differential Pressure Indicator Transmitter (pressure-drop/filtration resistance)",
}


def _safe_retry_after_seconds(response: Optional[requests.Response]) -> Optional[float]:
    if response is None:
        return None

    value = response.headers.get("Retry-After")
    if value is None:
        return None

    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None

    return max(0.0, seconds)


def _build_backoff_delay(attempt_index: int) -> float:
    # Exponential backoff with small jitter to avoid synchronized retries.
    base = STEP16_RETRY_DELAY * (2 ** attempt_index)
    jitter = random.uniform(0.0, 1.0)
    return base + jitter

def load_data():
    return (
        pd.read_csv(STEP12_PATH),
        pd.read_csv(STEP13_PATH),
        pd.read_csv(STEP14_PATH),
    )


def _write_csv_with_fallback(df: pd.DataFrame, path: str) -> str:
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot write output to {path}. Close the file if it is open and run again."
        ) from exc

# =========================
# GROQ LLM CALL
# =========================
def call_llm(prompt: str, session=None, model="llama-3.1-8b-instant", max_retries=3):
    url = "https://api.groq.com/openai/v1/chat/completions"
    api_key = GROQ_API_KEY

    if not api_key:
        return {
            "summary": "Groq failed: missing GROQ_API_KEY environment variable.",
            "root_cause_explanation": "Error",
            "propagation_explanation": "Error",
            "confidence_explanation": "Error",
            "recommendation": "Set GROQ_API_KEY and retry.",
        }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0
    }

    http = session or requests.Session()

    start_call = perf_counter()

    for attempt in range(max_retries):
        try:
            r = http.post(url, headers=headers, json=payload, timeout=60)
            if r.status_code in (429, 503):
                retry_after = _safe_retry_after_seconds(r)
                if attempt == max_retries - 1:
                    result = {
                        "summary": f"Groq rate-limited ({r.status_code}) after {max_retries} retries.",
                        "root_cause_explanation": "Not available due to API rate limits.",
                        "propagation_explanation": "Not available due to API rate limits.",
                        "confidence_explanation": "Rate limit exceeded; retry later or lower --llm-workers/--limit.",
                        "recommendation": "Wait and rerun with fewer concurrent requests.",
                    }
                    result["_latency_seconds"] = round(perf_counter() - start_call, 3)
                    return result

                sleep_seconds = retry_after if retry_after is not None else _build_backoff_delay(attempt)
                time.sleep(sleep_seconds)
                continue

            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
            parsed = _extract_json_object(text)
            result = _normalize_llm_explanation(parsed, text)
            result["_latency_seconds"] = round(perf_counter() - start_call, 3)
            return result
        except requests.RequestException as e:
            response = getattr(e, "response", None)
            status_code = response.status_code if response is not None else None

            if status_code in (429, 503):
                if attempt == max_retries - 1:
                    result = {
                        "summary": f"Groq rate-limited ({status_code}) after {max_retries} retries.",
                        "root_cause_explanation": "Not available due to API rate limits.",
                        "propagation_explanation": "Not available due to API rate limits.",
                        "confidence_explanation": "Rate limit exceeded; retry later or lower --llm-workers/--limit.",
                        "recommendation": "Wait and rerun with fewer concurrent requests.",
                    }
                    result["_latency_seconds"] = round(perf_counter() - start_call, 3)
                    return result

                retry_after = _safe_retry_after_seconds(response)
                sleep_seconds = retry_after if retry_after is not None else _build_backoff_delay(attempt)
                time.sleep(sleep_seconds)
                continue

            if attempt == max_retries - 1:
                result = {
                    "summary": f"Groq failed: {e}",
                    "root_cause_explanation": "Error",
                    "propagation_explanation": "Error",
                    "confidence_explanation": "Error",
                    "recommendation": "Retry",
                }
                result["_latency_seconds"] = round(perf_counter() - start_call, 3)
                return result

            time.sleep(_build_backoff_delay(attempt))
        except Exception as e:
            if attempt == max_retries - 1:
                result = {
                    "summary": f"Groq failed: {e}",
                    "root_cause_explanation": "Error",
                    "propagation_explanation": "Error",
                    "confidence_explanation": "Error",
                    "recommendation": "Retry",
                }
                result["_latency_seconds"] = round(perf_counter() - start_call, 3)
                return result

            time.sleep(_build_backoff_delay(attempt))


# =========================
# HELPERS
# =========================
def _load_structured_value(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (list, dict, tuple)):
        return value
    if not isinstance(value, str):
        return value

    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(value)
        except:
            pass
    return value


def normalize_bool(v):
    return str(v).lower() in {"true", "1", "yes"}


def clamp01(v):
    try:
        v = float(v)
    except:
        return 0.0
    return max(0.0, min(1.0, v))


def parse_candidate_paths(v):
    parsed = _load_structured_value(v)
    if isinstance(parsed, str):
        return [[x.strip() for x in parsed.split("->")]]
    if isinstance(parsed, list):
        return [parsed]
    return []


def format_path(nodes):
    if not nodes:
        return ""

    # Flatten if nested list
    if isinstance(nodes[0], list):
        nodes = nodes[0]

    return " -> ".join([str(n) for n in nodes if n])


def parse_sensor_list(value):
    parsed = _load_structured_value(value)
    if parsed is None:
        return []

    if isinstance(parsed, (list, tuple, set)):
        return [str(item).strip() for item in parsed if str(item).strip()]

    if isinstance(parsed, str):
        text = parsed.strip()
        if not text:
            return []
        if "->" in text:
            return [part.strip() for part in text.split("->") if part.strip()]
        if "," in text:
            return [part.strip() for part in text.split(",") if part.strip()]
        return [text]

    return [str(parsed).strip()]


def get_sensor_family(sensor_name: str) -> Optional[str]:
    match = re.match(r"([A-Za-z]+)", sensor_name or "")
    if not match:
        return None
    return match.group(1).upper()


def build_sensor_context(sensor_names: List[str]) -> List[str]:
    descriptions: List[str] = []
    seen_families = set()

    for sensor_name in sensor_names:
        family = get_sensor_family(sensor_name)
        if not family or family in seen_families:
            continue
        meaning = SENSOR_PREFIX_CONTEXT.get(family)
        if meaning:
            descriptions.append(f"{family}: {meaning}")
            seen_families.add(family)

    return descriptions


def build_stage_context(stage: str) -> Dict[str, Any]:
    return STAGE_CONTEXT.get(
        stage,
        {
            "name": "Unknown stage",
            "description": "No stage-specific context available.",
            "modeled_variables": [],
        },
    )


def parse_root_causes(value: Any) -> List[Dict[str, Any]]:
    parsed = _load_structured_value(value)
    if parsed is None:
        return []

    rows: List[Dict[str, Any]] = []
    if isinstance(parsed, dict):
        for sensor, score in parsed.items():
            try:
                rows.append({"sensor": str(sensor), "score": float(score)})
            except (TypeError, ValueError):
                continue
        return rows

    if isinstance(parsed, (list, tuple)):
        for item in parsed:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                sensor = str(item[0]).strip()
                if not sensor:
                    continue
                try:
                    score = float(item[1])
                except (TypeError, ValueError):
                    score = 0.0
                rows.append({"sensor": sensor, "score": score})
            elif isinstance(item, dict):
                sensor = str(item.get("sensor") or item.get("name") or item.get("root") or "").strip()
                if not sensor:
                    continue
                try:
                    score = float(item.get("score", 0.0))
                except (TypeError, ValueError):
                    score = 0.0
                rows.append({"sensor": sensor, "score": score})

    return rows

def merge_data(df12, df13, df14):
    df12["candidate_path"] = df12["propagation_paths"].apply(lambda x: format_path(parse_candidate_paths(x)[0]) if parse_candidate_paths(x) else "")

    df13["candidate_path"] = df13["candidate_path"].astype(str)
    df14["candidate_path"] = df14["candidate_path"].astype(str)

    merged = df12.merge(df13, on=["anomaly_id", "candidate_path"], how="left")
    merged = merged.merge(df14, on=["anomaly_id", "candidate_path"], how="left")

    merged["llm_confidence"] = merged["confidence"].apply(clamp01) if "confidence" in merged.columns else 0.0
    merged["temporal_score"] = merged["temporal_score"].apply(clamp01) if "temporal_score" in merged.columns else 0.0

    return merged


def _resolve_row_value(row: pd.Series, candidates: List[str], default: Any = None) -> Any:
    for key in candidates:
        if key in row and pd.notna(row[key]):
            return row[key]
    return default


def _extract_rca_score(row: pd.Series) -> float:
    root_causes_value = _resolve_row_value(row, ["root_causes", "root_causes_x", "root_causes_y"], default=[])
    root_causes = parse_root_causes(root_causes_value)
    if not root_causes:
        return 0.0
    return clamp01(max(item.get("score", 0.0) for item in root_causes))


def _temporal_priority(label_value: Any) -> int:
    label = str(label_value).strip().lower()
    return TEMPORAL_LABEL_PRIORITY.get(label, 0)


# =========================
# SCORING
# =========================
def compute_score(row):
    return (
        WEIGHT_LLM * row["llm_confidence"]
        + WEIGHT_TEMPORAL * row["temporal_score"]
        + WEIGHT_RCA * row["rca_score"]
    )


def compute_path_confidence_score(row):
    # Pre-LLM ranking score: use only temporal + RCA evidence.
    # Normalize by the sum of the active weights (0.4 + 0.2 = 0.6).
    temporal_component = WEIGHT_TEMPORAL * row["temporal_score"]
    rca_component = WEIGHT_RCA * row["rca_score"]
    return (temporal_component + rca_component) / (WEIGHT_TEMPORAL + WEIGHT_RCA)


def select_best_path(df):
    ranked_df = df.copy()

    ranked_df["rca_score"] = ranked_df.apply(_extract_rca_score, axis=1)
    ranked_df["path_confidence_score"] = ranked_df.apply(compute_path_confidence_score, axis=1)
    ranked_df["final_score"] = ranked_df.apply(compute_score, axis=1)

    if "path_valid" in ranked_df.columns:
        ranked_df["path_valid_rank"] = ranked_df["path_valid"].apply(normalize_bool).astype(int)
    else:
        ranked_df["path_valid_rank"] = 1

    temporal_label_series = ranked_df["temporal_label"] if "temporal_label" in ranked_df.columns else ""
    ranked_df["temporal_priority"] = temporal_label_series.apply(_temporal_priority)

    ranked_df = ranked_df.sort_values(
        by=[
            "path_valid_rank",
            "temporal_priority",
            "path_confidence_score",
            "final_score",
            "temporal_score",
            "llm_confidence",
            "rca_score",
        ],
        ascending=False,
    )

    return ranked_df.groupby("anomaly_id").head(1)


# =========================
# PROMPT + PARSING
# =========================
def build_prompt(row):
    anomaly_id = row.get("anomaly_id", "unknown")
    stage = str(row.get("stage", "unknown"))
    timestamp = row.get("timestamp", "unknown")

    candidate_path = str(row.get("candidate_path", ""))
    candidate_path_nodes = parse_sensor_list(candidate_path)
    anomalous_sensors = parse_sensor_list(row.get("anomalous_sensors", []))
    root_causes = parse_root_causes(row.get("root_causes", []))

    llm_confidence = clamp01(row.get("llm_confidence", 0.0))
    temporal_score = clamp01(row.get("temporal_score", 0.0))
    final_score = clamp01(row.get("final_score", 0.0))

    stage_context = build_stage_context(stage)
    sensor_context = build_sensor_context(anomalous_sensors + candidate_path_nodes)

    payload = {
        "anomaly_id": anomaly_id,
        "stage": stage,
        "stage_name": stage_context["name"],
        "timestamp": timestamp,
        "anomalous_sensors": anomalous_sensors,
        "root_causes": root_causes,
        "selected_path": candidate_path_nodes,
        "llm_confidence": llm_confidence,
        "temporal_score": temporal_score,
        "final_score": final_score,
        "stage_modeled_variables": stage_context["modeled_variables"],
    }

    return (
        "You are an industrial anomaly analysis assistant for SWaT (Secure Water Treatment). "
        "Given RCA output and a selected propagation path, explain the anomaly clearly and technically.\n\n"
        f"Dataset context: {SWAT_DATASET_CONTEXT}\n"
        f"Stage context: {stage} is {stage_context['name']}. {stage_context['description']}\n"
        "In this repository, explanations should focus on continuous .Pv process-value sensors.\n"
        f"Modeled variables for this stage: {', '.join(stage_context['modeled_variables']) if stage_context['modeled_variables'] else 'unknown'}\n"
        f"Sensor tag meanings in this case: {'; '.join(sensor_context) if sensor_context else 'no additional tag context available'}\n\n"
        "Write concise, evidence-based reasoning. Avoid speculation beyond the provided data.\n"
        "Use the selected path as the main propagation narrative and relate it to anomalous sensors and top root-cause scores.\n"
        "When confidence is limited (low llm_confidence or temporal_score), explicitly say uncertainty is moderate/high.\n\n"
        "Return only valid JSON with this exact schema:\n"
        "{\n"
        '  "summary": "...",\n'
        '  "root_cause_explanation": "...",\n'
        '  "propagation_explanation": "...",\n'
        '  "confidence_explanation": "...",\n'
        '  "recommendation": "..."\n'
        "}\n\n"
        f"Case data:\n{json.dumps(payload, ensure_ascii=True)}"
    )


def _extract_json_object(text):
    try:
        return json.loads(text)
    except:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                pass
    return {}


def _normalize_llm_explanation(payload, fallback):
    if not payload:
        return {
            "summary": fallback,
            "root_cause_explanation": "",
            "propagation_explanation": "",
            "confidence_explanation": "",
            "recommendation": "",
        }
    return payload


def _first_present_value(row: pd.Series, candidates: List[str], default: Any = "") -> Any:
    for key in candidates:
        if key in row and pd.notna(row[key]):
            value = row[key]
            if isinstance(value, str) and not value.strip():
                continue
            return value
    return default


def _build_compact_output_row(row: pd.Series, llm_payload: Dict[str, str]) -> Dict[str, Any]:
    return {
        "anomaly_id": _first_present_value(row, ["anomaly_id"], ""),
        "stage": _first_present_value(row, ["stage", "stage_x", "stage_y"], ""),
        "timestamp": _first_present_value(row, ["timestamp", "timestamp_x", "timestamp_y", "t_stamp"], ""),
        "best_path": _first_present_value(row, ["candidate_path"], ""),
        "path_confidence_score": clamp01(_first_present_value(row, ["path_confidence_score"], 0.0)),
        "llm_confidence": clamp01(_first_present_value(row, ["llm_confidence", "confidence"], 0.0)),
        "temporal_score": clamp01(_first_present_value(row, ["temporal_score"], 0.0)),
        "rca_score": clamp01(_first_present_value(row, ["rca_score"], 0.0)),
        "final_score": clamp01(_first_present_value(row, ["final_score"], 0.0)),
        "summary": str(llm_payload.get("summary", "")).strip(),
        "root_cause_explanation": str(llm_payload.get("root_cause_explanation", "")).strip(),
        "propagation_explanation": str(llm_payload.get("propagation_explanation", "")).strip(),
        "confidence_explanation": str(llm_payload.get("confidence_explanation", "")).strip(),
        "recommendation": str(llm_payload.get("recommendation", "")).strip(),
    }


# =========================
# MAIN PIPELINE
# =========================
def run_step16(limit=0, llm_workers=1, log_every=1):

    t0 = perf_counter()
    print("[STEP16] Starting run")

    t_load_start = perf_counter()
    df12, df13, df14 = load_data()
    print(f"[STEP16] Load data: {perf_counter() - t_load_start:.2f}s | step12={len(df12)} step13={len(df13)} step14={len(df14)}")

    t_merge_start = perf_counter()
    merged = merge_data(df12, df13, df14)
    print(f"[STEP16] Merge data: {perf_counter() - t_merge_start:.2f}s | merged_rows={len(merged)}")

    t_select_start = perf_counter()
    selected = select_best_path(merged)
    print(f"[STEP16] Select best path: {perf_counter() - t_select_start:.2f}s | selected_rows={len(selected)}")

    if limit > 0:
        selected = selected.head(limit)
        print(f"[LIMIT] Running {len(selected)} anomalies")

    results = []

    def process(row):
        prompt = build_prompt(row)
        return call_llm(prompt)

    llm_latencies: List[float] = []
    llm_phase_start = perf_counter()

    if llm_workers > 1:
        with ThreadPoolExecutor(llm_workers) as ex:
            futures = [ex.submit(process, row) for _, row in selected.iterrows()]
            for idx, (f, (_, row)) in enumerate(zip(futures, selected.iterrows()), start=1):
                res = f.result()
                llm_latencies.append(float(res.get("_latency_seconds", 0.0) or 0.0))
                results.append(_build_compact_output_row(row, res))
                if log_every > 0 and (idx % log_every == 0 or idx == len(selected)):
                    avg = (sum(llm_latencies) / len(llm_latencies)) if llm_latencies else 0.0
                    print(f"[STEP16] Progress {idx}/{len(selected)} | last_llm={llm_latencies[-1]:.2f}s | avg_llm={avg:.2f}s")
    else:
        for idx, (_, row) in enumerate(selected.iterrows(), start=1):
            res = process(row)
            llm_latencies.append(float(res.get("_latency_seconds", 0.0) or 0.0))
            results.append(_build_compact_output_row(row, res))
            if log_every > 0 and (idx % log_every == 0 or idx == len(selected)):
                avg = (sum(llm_latencies) / len(llm_latencies)) if llm_latencies else 0.0
                print(f"[STEP16] Progress {idx}/{len(selected)} | last_llm={llm_latencies[-1]:.2f}s | avg_llm={avg:.2f}s")

    llm_phase_elapsed = perf_counter() - llm_phase_start
    avg_llm = (sum(llm_latencies) / len(llm_latencies)) if llm_latencies else 0.0
    print(f"[STEP16] LLM phase: {llm_phase_elapsed:.2f}s | calls={len(llm_latencies)} | avg_per_call={avg_llm:.2f}s")

    out = pd.DataFrame(results)
    for col in OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    out = out[OUTPUT_COLUMNS].copy()
    actual_output_path = _write_csv_with_fallback(out, OUTPUT_PATH)

    print(f"Saved to {actual_output_path}")
    print(f"[STEP16] Total runtime: {perf_counter() - t0:.2f}s")
    return out


# =========================
# ARGPARSE
# =========================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=len(pd.read_csv(STEP12_PATH)), help="Limit number of anomalies to process (for testing)")
    parser.add_argument("--llm-workers", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=1, help="Log progress every N anomalies")
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    run_step16(limit=args.limit, llm_workers=args.llm_workers, log_every=args.log_every)