import argparse
import ast
import json
import os
import random
import re
import time
from datetime import datetime
from time import perf_counter
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

# Project-relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

os.makedirs(os.path.join(PROJECT_ROOT, "data", "processed", "step16"), exist_ok=True)

STEP12_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "step12", "swat_rca_step12_results.csv")
STEP13_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "step13", "all_propagation_path_llm_evaluation_20260329_040216.csv")
STEP14_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "step14", "propagation_path_temporal_evaluation.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "step16", "all_llm_explanations_gpu.csv")

OUTPUT_COLUMNS = [
    "anomaly_id", "stage", "timestamp", "best_path", "path_confidence_score", "llm_confidence", "temporal_score", "rca_score", "final_score", "summary", "root_cause_explanation", "propagation_explanation", "confidence_explanation", "recommendation"
]

# ========== DATA LOADERS ==========
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
        raise PermissionError(f"Cannot write output to {path}. Close the file if it is open and run again.") from exc

# ========== PARSING HELPERS ==========
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
    merged = df13.copy()
    merged = merged.merge(df12, on=["anomaly_id", "candidate_path"], how="left", suffixes=("", "_step12"))
    merged = merged.merge(df14, on=["anomaly_id", "candidate_path"], how="left", suffixes=("", "_step14"))
    merged["llm_confidence"] = merged["confidence"].apply(clamp01) if "confidence" in merged.columns else 0.0
    merged["temporal_score"] = merged["temporal_score"].apply(clamp01) if "temporal_score" in merged.columns else 0.0
    return merged

def _extract_rca_score(row: pd.Series) -> float:
    root_causes_value = row.get("root_causes", [])
    root_causes = parse_root_causes(root_causes_value)
    if not root_causes:
        return 0.0
    return clamp01(max(item.get("score", 0.0) for item in root_causes))

def compute_path_confidence_score(row):
    temporal_component = 0.4 * row["temporal_score"]
    rca_component = 0.2 * row["rca_score"]
    return (temporal_component + rca_component) / 0.6

def compute_score(row):
    return 0.4 * row["llm_confidence"] + 0.4 * row["temporal_score"] + 0.2 * row["rca_score"]

def select_best_path(df):
    ranked_df = df.copy()
    ranked_df["rca_score"] = ranked_df.apply(_extract_rca_score, axis=1)
    ranked_df["path_confidence_score"] = ranked_df.apply(compute_path_confidence_score, axis=1)
    ranked_df["final_score"] = ranked_df.apply(compute_score, axis=1)
    ranked_df = ranked_df.sort_values(
        by=["path_confidence_score", "final_score", "temporal_score", "llm_confidence", "rca_score"],
        ascending=False,
    )
    return ranked_df.groupby("anomaly_id").head(1)

# ========== OLLAMA LLM CALL ==========
def call_ollama(prompt: str, model="qwen2.5:7b", max_retries=3, base_url="http://localhost:11434"):
    url = f"{base_url}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }
    for attempt in range(max_retries):
        try:
            r = requests.post(url, json=payload, timeout=120)
            r.raise_for_status()
            text = r.json()["message"]["content"]
            return _extract_json_object(text)
        except Exception as e:
            if attempt == max_retries - 1:
                return {
                    "summary": f"Ollama failed: {e}",
                    "root_cause_explanation": "Error",
                    "propagation_explanation": "Error",
                    "confidence_explanation": "Error",
                    "recommendation": "Retry",
                }
            time.sleep(2 ** attempt)

# ========== PROMPT BUILDER ==========
def build_prompt(row):
    anomaly_id = row.get("anomaly_id", "unknown")
    stage = str(row.get("stage", "unknown"))
    timestamp = row.get("timestamp", "unknown")
    candidate_path = str(row.get("candidate_path", ""))
    anomalous_sensors = parse_sensor_list(row.get("anomalous_sensors", []))
    root_causes = parse_root_causes(row.get("root_causes", []))
    llm_confidence = clamp01(row.get("llm_confidence", 0.0))
    temporal_score = clamp01(row.get("temporal_score", 0.0))
    final_score = clamp01(row.get("final_score", 0.0))
    payload = {
        "anomaly_id": anomaly_id,
        "stage": stage,
        "timestamp": timestamp,
        "anomalous_sensors": anomalous_sensors,
        "root_causes": root_causes,
        "selected_path": candidate_path,
        "llm_confidence": llm_confidence,
        "temporal_score": temporal_score,
        "final_score": final_score,
    }
    return (
        "You are an industrial anomaly analysis assistant for SWaT (Secure Water Treatment). "
        "Given RCA output and a selected propagation path, explain the anomaly clearly and technically.\n\n"
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

def _build_output_row(row: pd.Series, llm_payload: Dict[str, str]) -> Dict[str, Any]:
    return {
        "anomaly_id": row.get("anomaly_id", ""),
        "stage": row.get("stage", ""),
        "timestamp": row.get("timestamp", ""),
        "best_path": row.get("candidate_path", ""),
        "path_confidence_score": clamp01(row.get("path_confidence_score", 0.0)),
        "llm_confidence": clamp01(row.get("llm_confidence", 0.0)),
        "temporal_score": clamp01(row.get("temporal_score", 0.0)),
        "rca_score": clamp01(row.get("rca_score", 0.0)),
        "final_score": clamp01(row.get("final_score", 0.0)),
        "summary": str(llm_payload.get("summary", "")).strip(),
        "root_cause_explanation": str(llm_payload.get("root_cause_explanation", "")).strip(),
        "propagation_explanation": str(llm_payload.get("propagation_explanation", "")).strip(),
        "confidence_explanation": str(llm_payload.get("confidence_explanation", "")).strip(),
        "recommendation": str(llm_payload.get("recommendation", "")).strip(),
    }

# ========== MAIN PIPELINE ==========
def run_step16_gpu(limit=0, log_every=1000):
    t0 = perf_counter()
    print("[STEP16-GPU] Starting run")
    try:
        df12, df13, df14 = load_data()
        print(f"[STEP16-GPU] Loaded data: step12={len(df12)}, step13={len(df13)}, step14={len(df14)}")
        merged = merge_data(df12, df13, df14)
        print(f"[STEP16-GPU] Merged data: {len(merged)} rows")
        best_paths = select_best_path(merged)
        print(f"[STEP16-GPU] Selected best paths: {len(best_paths)} anomalies")
        if limit > 0:
            best_paths = best_paths.head(limit)
            print(f"[STEP16-GPU] Limiting to {len(best_paths)} anomalies")
        results = []
        for idx, (_, row) in enumerate(best_paths.iterrows(), start=1):
            try:
                prompt = build_prompt(row)
                llm_payload = call_ollama(prompt)
                results.append(_build_output_row(row, llm_payload))
            except Exception as e:
                print(f"[STEP16-GPU][ERROR] at anomaly {row.get('anomaly_id', idx)}: {e}")
                continue
            if idx % 25 == 0:
                print(f"[STEP16-GPU] Processing: {idx}/{len(best_paths)} anomalies completed")
            if idx % log_every == 0 or idx == len(best_paths):
                out = pd.DataFrame(results)
                for col in OUTPUT_COLUMNS:
                    if col not in out.columns:
                        out[col] = ""
                out = out[OUTPUT_COLUMNS].copy()
                _write_csv_with_fallback(out, OUTPUT_PATH)
                print(f"[STEP16-GPU] Progress: {idx}/{len(best_paths)} saved to {OUTPUT_PATH}")
        out = pd.DataFrame(results)
        for col in OUTPUT_COLUMNS:
            if col not in out.columns:
                out[col] = ""
        out = out[OUTPUT_COLUMNS].copy()
        _write_csv_with_fallback(out, OUTPUT_PATH)
        print(f"[STEP16-GPU] Done. Saved to {OUTPUT_PATH}")
        print(f"[STEP16-GPU] Total runtime: {perf_counter() - t0:.2f}s")
        return out
    except Exception as e:
        print(f"[STEP16-GPU][FATAL ERROR]: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Limit number of anomalies to process (for testing)")
    parser.add_argument("--log-every", type=int, default=1000, help="Save output every N anomalies")
    args = parser.parse_args()
    run_step16_gpu(limit=args.limit, log_every=args.log_every)