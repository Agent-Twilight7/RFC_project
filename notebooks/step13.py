# """
# Step 13: LLM-based propagation path evaluation

# Pipeline:
# 1. Load candidate propagation paths directly from a CSV, or derive them from Step 12 outputs.
# 2. Build a prompt for each anomaly/path pair.
# 3. Call a local Ollama model to judge path plausibility.
# 4. Parse the JSON response.
# 5. Save row-level path evaluation results to CSV.
# """

# import argparse
# import ast
# import json
# import os
# import re
# import time
# from typing import Any, Dict, List, Optional

# import pandas as pd
# import requests


# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# STEP12_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "step12")
# STEP13_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "step13")
# os.makedirs(STEP13_DIR, exist_ok=True)

# DEFAULT_INPUT_PATH = os.path.join(STEP12_DIR, "swat_rca_step12_results.csv")
# DEFAULT_OUTPUT_PATH = os.path.join(STEP13_DIR, "step_13_propagation_path_llm_evaluation.csv")

# OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
# OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
# OLLAMA_TIMEOUT = int(os.getenv("STEP13_OLLAMA_TIMEOUT", "120"))
# STEP13_OLLAMA_NUM_GPU = int(os.getenv("STEP13_OLLAMA_NUM_GPU", "-1"))
# STEP13_OLLAMA_MAIN_GPU = os.getenv("STEP13_OLLAMA_MAIN_GPU")
# STEP13_PROGRESS_EVERY = int(os.getenv("STEP13_PROGRESS_EVERY", "25"))
# STEP13_MAX_RETRIES = int(os.getenv("STEP13_MAX_RETRIES", "3"))
# STEP13_RETRY_DELAY = float(os.getenv("STEP13_RETRY_DELAY", "2"))

# if STEP13_OLLAMA_MAIN_GPU is not None and STEP13_OLLAMA_MAIN_GPU != "":
#     STEP13_OLLAMA_MAIN_GPU = int(STEP13_OLLAMA_MAIN_GPU)
# else:
#     STEP13_OLLAMA_MAIN_GPU = None

# SWAT_DATASET_CONTEXT = (
#     "SWaT stands for Secure Water Treatment. It is a cyber-physical industrial water treatment "
#     "dataset with time-series measurements from sensors and actuators across six process stages. "
#     "In this repository, anomaly detection is performed on continuous process-value sensors ending in .Pv, "
#     "and candidate propagation paths come from Bayesian Network root-cause analysis."
# )

# STAGE_CONTEXT = {
#     "P1": {
#         "name": "Raw Water Intake",
#         "description": "Intake-side stage focused on incoming water level and inlet flow behavior.",
#         "modeled_variables": ["LIT101.Pv", "FIT101.Pv"],
#     },
#     "P2": {
#         "name": "Pre-treatment",
#         "description": "Pre-treatment stage with one flow signal and multiple analyzer measurements.",
#         "modeled_variables": ["FIT201.Pv", "AIT201.Pv", "AIT202.Pv", "AIT203.Pv"],
#     },
#     "P3": {
#         "name": "Ultra-Filtration",
#         "description": "Ultra-filtration stage combining analyzer, level, flow, and differential pressure signals.",
#         "modeled_variables": ["AIT301.Pv", "AIT302.Pv", "AIT303.Pv", "LIT301.Pv", "FIT301.Pv", "DPIT301.Pv"],
#     },
#     "P4": {
#         "name": "De-Chlorination",
#         "description": "De-chlorination stage with analyzer, level, flow, and pressure measurements.",
#         "modeled_variables": ["AIT401.Pv", "AIT402.Pv", "LIT401.Pv", "FIT401.Pv"],
#     },
#     "P5": {
#         "name": "Reverse Osmosis",
#         "description": "Reverse-osmosis stage with analyzer, flow, and pressure behavior across membrane-related equipment.",
#         "modeled_variables": [
#             "AIT501.Pv",
#             "AIT502.Pv",
#             "AIT503.Pv",
#             "AIT504.Pv",
#             "FIT501.Pv",
#             "FIT502.Pv",
#             "FIT503.Pv",
#             "FIT504.Pv",
#             "PIT501.Pv",
#             "PIT502.Pv",
#             "PIT503.Pv",
#         ],
#     },
#     "P6": {
#         "name": "Disposition",
#         "description": "Final disposition stage with a very small continuous measurement set in this repository.",
#         "modeled_variables": ["FIT601.Pv"],
#     },
# }

# SENSOR_PREFIX_CONTEXT = {
#     "FIT": "flow indicator transmitter, usually a flow-related measurement",
#     "LIT": "level indicator transmitter, usually a tank or vessel level measurement",
#     "AIT": "analyzer indicator transmitter, usually a water quality or chemistry measurement",
#     "PIT": "pressure indicator transmitter, usually a pressure measurement",
#     "DPIT": "differential pressure indicator transmitter, usually a pressure-drop or filtration-resistance measurement",
# }


# def _load_structured_value(value: Any) -> Any:
#     if value is None:
#         return None
#     if isinstance(value, float) and pd.isna(value):
#         return None
#     if isinstance(value, (list, dict, tuple)):
#         return value
#     if not isinstance(value, str):
#         return value

#     text = value.strip()
#     if not text:
#         return None

#     for parser in (json.loads, ast.literal_eval):
#         try:
#             return parser(text)
#         except (json.JSONDecodeError, ValueError, SyntaxError):
#             continue

#     return text


# def parse_sensor_list(value: Any) -> List[str]:
#     parsed = _load_structured_value(value)
#     if parsed is None:
#         return []

#     if isinstance(parsed, (list, tuple, set)):
#         return [str(item).strip() for item in parsed if str(item).strip()]

#     if isinstance(parsed, str):
#         if "->" in parsed:
#             return [part.strip() for part in parsed.split("->") if part.strip()]
#         if "," in parsed:
#             return [part.strip() for part in parsed.split(",") if part.strip()]
#         return [parsed.strip()] if parsed.strip() else []

#     return [str(parsed).strip()]


# def parse_candidate_paths(value: Any) -> List[List[str]]:
#     parsed = _load_structured_value(value)
#     if parsed is None:
#         return []

#     if isinstance(parsed, str):
#         nodes = [part.strip() for part in parsed.split("->") if part.strip()]
#         return [nodes] if nodes else []

#     if isinstance(parsed, (list, tuple)):
#         if not parsed:
#             return []

#         first = parsed[0]
#         if isinstance(first, (list, tuple)):
#             return [
#                 [str(node).strip() for node in path if str(node).strip()]
#                 for path in parsed
#                 if path
#             ]

#         return [[str(node).strip() for node in parsed if str(node).strip()]]

#     return [[str(parsed).strip()]]


# def format_path(path_nodes: List[str]) -> str:
#     return " -> ".join(path_nodes)


# def get_sensor_family(sensor_name: str) -> Optional[str]:
#     match = re.match(r"([A-Za-z]+)", sensor_name or "")
#     if not match:
#         return None
#     return match.group(1).upper()


# def build_sensor_context(sensor_names: List[str]) -> List[str]:
#     descriptions: List[str] = []
#     seen_families = set()

#     for sensor_name in sensor_names:
#         family = get_sensor_family(sensor_name)
#         if not family or family in seen_families:
#             continue
#         meaning = SENSOR_PREFIX_CONTEXT.get(family)
#         if meaning:
#             descriptions.append(f"{family}: {meaning}")
#             seen_families.add(family)

#     return descriptions


# def build_stage_context(stage: str) -> Dict[str, Any]:
#     return STAGE_CONTEXT.get(
#         stage,
#         {
#             "name": "Unknown stage",
#             "description": "No stage-specific context available.",
#             "modeled_variables": [],
#         },
#     )


# def normalize_input_rows(df: pd.DataFrame) -> pd.DataFrame:
#     if {"anomaly_sensors", "candidate_path"}.issubset(df.columns):
#         normalized = df.copy()
#         normalized["anomaly_sensors"] = normalized["anomaly_sensors"].apply(parse_sensor_list)
#         normalized["candidate_path_nodes"] = normalized["candidate_path"].apply(
#             lambda value: parse_candidate_paths(value)[0] if parse_candidate_paths(value) else []
#         )
#         normalized["candidate_path"] = normalized["candidate_path_nodes"].apply(format_path)
#         return normalized

#     if {"anomalous_sensors", "propagation_paths"}.issubset(df.columns):
#         expanded_rows: List[Dict[str, Any]] = []

#         for _, row in df.iterrows():
#             row_dict = row.to_dict()
#             anomaly_sensors = parse_sensor_list(row_dict.get("anomalous_sensors"))
#             candidate_paths = parse_candidate_paths(row_dict.get("propagation_paths"))

#             for path_index, path_nodes in enumerate(candidate_paths, start=1):
#                 expanded_row = dict(row_dict)
#                 expanded_row["anomaly_sensors"] = anomaly_sensors
#                 expanded_row["candidate_path_index"] = path_index
#                 expanded_row["candidate_path_nodes"] = path_nodes
#                 expanded_row["candidate_path"] = format_path(path_nodes)
#                 expanded_rows.append(expanded_row)

#         return pd.DataFrame(expanded_rows)

#     raise ValueError(
#         "Input CSV must contain either 'anomaly_sensors' and 'candidate_path', "
#         "or Step 12 columns 'anomalous_sensors' and 'propagation_paths'."
#     )


# def build_prompt(row: Dict[str, Any]) -> str:
#     anomaly_id = row.get("anomaly_id", "unknown")
#     stage = row.get("stage", "unknown")
#     phase = row.get("phase", "unknown")
#     timestamp = row.get("timestamp", "unknown")
#     sensors = row.get("anomaly_sensors", [])
#     path_nodes = row.get("candidate_path_nodes", [])
#     stage_context = build_stage_context(stage)
#     sensor_context = build_sensor_context(sensors + path_nodes)

#     payload = {
#         "anomaly_id": anomaly_id,
#         "stage": stage,
#         "stage_name": stage_context["name"],
#         "phase": phase,
#         "timestamp": timestamp,
#         "anomaly_sensors": sensors,
#         "candidate_path": path_nodes,
#         "stage_modeled_variables": stage_context["modeled_variables"],
#     }

#     return (
#         "You are evaluating anomaly propagation paths for the SWaT industrial control system. "
#         "Assess whether the candidate path is a plausible explanation for the observed anomaly.\n\n"
#         f"Dataset context: {SWAT_DATASET_CONTEXT}\n"
#         f"Stage context: {stage} is {stage_context['name']}. {stage_context['description']}\n"
#         "The stage-wise model in this repository uses only continuous .Pv process values, so the anomaly sensors and path nodes are continuous measurements rather than actuator states or alarm flags.\n"
#         f"Modeled variables for this stage: {', '.join(stage_context['modeled_variables']) if stage_context['modeled_variables'] else 'unknown'}\n"
#         f"Sensor tag meanings in this case: {'; '.join(sensor_context) if sensor_context else 'no additional tag context available'}\n\n"
#         "Evaluate exactly these criteria:\n"
#         "1. Sensor coverage: does the path cover the anomalous sensors, fully or partially?\n"
#         "2. Causal direction plausibility: does the path order look like a realistic propagation direction?\n"
#         "3. Propagation completeness: does the path form a coherent spread instead of skipping key sensors?\n\n"
#         "Guidance:\n"
#         "- Be conservative. Mark invalid if the path clearly fails to explain the anomaly sensors.\n"
#         "- A path can still be valid if it only partially covers the anomaly sensors but remains causally plausible.\n"
#         "- Prefer explanations that stay within the same stage and align with the measurement types involved.\n"
#         "- Treat flow, level, analyzer, pressure, and differential pressure signals as different but related physical measurements; use that to judge whether the path order looks coherent.\n"
#         "- Use the phase label only as supporting context, not as the main reason to accept or reject a path.\n"
#         "- Confidence must be a number between 0 and 1.\n"
#         "- Reason must be short and specific.\n\n"
#         "Return only valid JSON with this exact schema:\n"
#         "{\n"
#         '  "valid": true,\n'
#         '  "confidence": 0.0,\n'
#         '  "reason": "short explanation"\n'
#         "}\n\n"
#         f"Case data:\n{json.dumps(payload, ensure_ascii=True)}"
#     )


# def extract_json_object(text: str) -> Dict[str, Any]:
#     cleaned = text.strip()
#     if cleaned.startswith("```"):
#         cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
#         cleaned = re.sub(r"```$", "", cleaned).strip()

#     try:
#         return json.loads(cleaned)
#     except json.JSONDecodeError:
#         match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
#         if not match:
#             raise ValueError(f"No JSON object found in model response: {text}")
#         return json.loads(match.group(0))


# def normalize_llm_result(result: Dict[str, Any]) -> Dict[str, Any]:
#     valid = result.get("valid", False)
#     if isinstance(valid, str):
#         valid = valid.strip().lower() == "true"
#     else:
#         valid = bool(valid)

#     confidence = result.get("confidence", 0.0)
#     try:
#         confidence = float(confidence)
#     except (TypeError, ValueError):
#         confidence = 0.0
#     confidence = max(0.0, min(1.0, confidence))

#     reason = str(result.get("reason", "No explanation returned.")).strip()
#     if not reason:
#         reason = "No explanation returned."

#     return {
#         "valid": valid,
#         "confidence": confidence,
#         "reason": reason,
#     }


# def infer_ollama_api_base(generate_url: str) -> str:
#     if "/api/generate" in generate_url:
#         return generate_url.split("/api/generate", 1)[0]
#     if generate_url.endswith("/"):
#         return generate_url[:-1]
#     return generate_url


# def _extract_model_runtime_entry(ps_payload: Dict[str, Any], model_name: str) -> Optional[Dict[str, Any]]:
#     models = ps_payload.get("models")
#     if not isinstance(models, list):
#         return None

#     requested = (model_name or "").strip().lower()
#     requested_no_tag = requested.split(":", 1)[0]

#     for item in models:
#         if not isinstance(item, dict):
#             continue
#         candidate_names = [
#             str(item.get("name", "")).strip().lower(),
#             str(item.get("model", "")).strip().lower(),
#         ]
#         if requested in candidate_names:
#             return item

#         for candidate in candidate_names:
#             if candidate and candidate.split(":", 1)[0] == requested_no_tag:
#                 return item

#     return None


# def print_ollama_gpu_preflight(
#     model: str,
#     url: str,
#     num_gpu: int,
#     main_gpu: Optional[int],
# ) -> None:
#     if num_gpu == 0:
#         print("[STEP 13] WARNING: --num-gpu=0 forces CPU-only inference.")
#         return

#     base_url = infer_ollama_api_base(url)
#     ps_url = f"{base_url}/api/ps"

#     try:
#         response = requests.get(ps_url, timeout=5)
#         response.raise_for_status()
#         entry = _extract_model_runtime_entry(response.json(), model)

#         if entry is None:
#             print(
#                 "[STEP 13] Preflight: model not currently loaded in Ollama runtime; "
#                 "GPU status will be visible after first generation call."
#             )
#             return

#         size_vram = entry.get("size_vram")
#         if isinstance(size_vram, (int, float)) and size_vram > 0:
#             print(f"[STEP 13] Preflight: model appears GPU-backed (size_vram={int(size_vram)} bytes).")
#             return

#         if isinstance(size_vram, (int, float)) and size_vram == 0:
#             print(
#                 "[STEP 13] WARNING: loaded model reports size_vram=0, which suggests CPU-only runtime. "
#                 "Check Ollama CUDA setup if this is unexpected."
#             )
#             return

#         print(
#             "[STEP 13] Preflight: unable to infer GPU status from /api/ps payload. "
#             "Proceeding with requested Ollama options."
#         )
#     except requests.RequestException as exc:
#         print(f"[STEP 13] Preflight: could not query Ollama runtime status ({exc}).")
#         if num_gpu > 0 or num_gpu < 0 or main_gpu is not None:
#             print("[STEP 13] Preflight: continuing with GPU-enabled options request.")


# def print_ollama_gpu_runtime_status(model: str, url: str) -> None:
#     base_url = infer_ollama_api_base(url)
#     ps_url = f"{base_url}/api/ps"

#     try:
#         response = requests.get(ps_url, timeout=5)
#         response.raise_for_status()
#         entry = _extract_model_runtime_entry(response.json(), model)

#         if entry is None:
#             print("[STEP 13] Runtime check: model still not listed in /api/ps.")
#             return

#         size_vram = entry.get("size_vram")
#         if isinstance(size_vram, (int, float)):
#             if size_vram > 0:
#                 print(f"[STEP 13] Runtime check: GPU confirmed (size_vram={int(size_vram)} bytes).")
#             else:
#                 print("[STEP 13] Runtime check: likely CPU-only (size_vram=0).")
#             return

#         print("[STEP 13] Runtime check: GPU status unavailable (missing size_vram).")
#     except requests.RequestException as exc:
#         print(f"[STEP 13] Runtime check: could not query /api/ps ({exc}).")


# def call_ollama(
#     prompt: str,
#     session: requests.Session,
#     model: str = OLLAMA_MODEL,
#     url: str = OLLAMA_URL,
#     timeout: int = OLLAMA_TIMEOUT,
#     num_gpu: int = STEP13_OLLAMA_NUM_GPU,
#     main_gpu: Optional[int] = STEP13_OLLAMA_MAIN_GPU,
#     max_retries: int = STEP13_MAX_RETRIES,
# ) -> Dict[str, Any]:
#     payload = {
#         "model": model,
#         "prompt": prompt,
#         "stream": False,
#         "format": "json",
#         "options": {
#             "temperature": 0.1,
#             "num_gpu": num_gpu,
#         },
#     }

#     if main_gpu is not None:
#         payload["options"]["main_gpu"] = main_gpu

#     last_error: Optional[Exception] = None

#     for attempt in range(1, max_retries + 1):
#         try:
#             response = session.post(url, json=payload, timeout=timeout)
#             response.raise_for_status()
#             body = response.json()
#             model_text = body.get("response", "")
#             if not model_text:
#                 raise ValueError(f"Ollama response is missing 'response': {body}")
#             return normalize_llm_result(extract_json_object(model_text))
#         except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
#             last_error = exc
#             if attempt == max_retries:
#                 break
#             time.sleep(STEP13_RETRY_DELAY * attempt)

#     raise RuntimeError(f"Failed to evaluate path with Ollama after {max_retries} attempts: {last_error}")


# def evaluate_dataset_rows(
#     df: pd.DataFrame,
#     model: str = OLLAMA_MODEL,
#     url: str = OLLAMA_URL,
#     timeout: int = OLLAMA_TIMEOUT,
#     num_gpu: int = STEP13_OLLAMA_NUM_GPU,
#     main_gpu: Optional[int] = STEP13_OLLAMA_MAIN_GPU,
#     max_rows: Optional[int] = None,
# ) -> pd.DataFrame:
#     normalized_df = normalize_input_rows(df)
#     if normalized_df.empty:
#         return normalized_df

#     if max_rows is not None:
#         normalized_df = normalized_df.head(max_rows).copy()

#     results: List[Dict[str, Any]] = []
#     total_rows = len(normalized_df)
#     runtime_status_printed = False

#     with requests.Session() as session:
#         for index, row in enumerate(normalized_df.to_dict(orient="records"), start=1):
#             try:
#                 evaluation = call_ollama(
#                     prompt=build_prompt(row),
#                     session=session,
#                     model=model,
#                     url=url,
#                     timeout=timeout,
#                     num_gpu=num_gpu,
#                     main_gpu=main_gpu,
#                 )
#                 if not runtime_status_printed:
#                     print_ollama_gpu_runtime_status(model=model, url=url)
#                     runtime_status_printed = True
#             except RuntimeError as exc:
#                 evaluation = {
#                     "valid": False,
#                     "confidence": 0.0,
#                     "reason": str(exc),
#                 }

#             result_row = dict(row)
#             result_row["path_valid"] = evaluation["valid"]
#             result_row["confidence"] = evaluation["confidence"]
#             result_row["reason"] = evaluation["reason"]

#             if isinstance(result_row.get("anomaly_sensors"), list):
#                 result_row["anomaly_sensors"] = json.dumps(result_row["anomaly_sensors"])
#             if isinstance(result_row.get("candidate_path_nodes"), list):
#                 result_row["candidate_path_nodes"] = json.dumps(result_row["candidate_path_nodes"])

#             results.append(result_row)

#             if index % max(1, STEP13_PROGRESS_EVERY) == 0 or index == total_rows:
#                 print(f"[STEP 13] Progress: {index}/{total_rows} paths evaluated")


#     return pd.DataFrame(results)


# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(description="Evaluate anomaly propagation paths with a local Ollama LLM.")
#     parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="Input CSV path.")
#     parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Output CSV path.")
#     parser.add_argument("--model", default=OLLAMA_MODEL, help="Ollama model name.")
#     parser.add_argument("--url", default=OLLAMA_URL, help="Ollama generate API URL.")
#     parser.add_argument("--timeout", type=int, default=OLLAMA_TIMEOUT, help="HTTP timeout in seconds.")
#     parser.add_argument(
#         "--num-gpu",
#         type=int,
#         default=STEP13_OLLAMA_NUM_GPU,
#         help="Ollama num_gpu option. Use 0 to force CPU, positive values to offload layers, -1 for Ollama default.",
#     )
#     parser.add_argument(
#         "--main-gpu",
#         type=int,
#         default=STEP13_OLLAMA_MAIN_GPU,
#         help="Optional Ollama main_gpu index (for multi-GPU systems).",
#     )
#     parser.add_argument("--max-rows", type=int, default=None, help="Optional row limit for quick runs.")
#     return parser.parse_args()


# def main() -> None:
#     args = parse_args()

#     print(f"[STEP 13] Loading input CSV: {args.input}")
#     df = pd.read_csv(args.input)
#     print(f"[STEP 13] Loaded {len(df)} source rows")
#     print(
#         f"[STEP 13] Ollama runtime options: model={args.model}, num_gpu={args.num_gpu}, "
#         f"main_gpu={args.main_gpu if args.main_gpu is not None else 'auto'}"
#     )
#     print_ollama_gpu_preflight(
#         model=args.model,
#         url=args.url,
#         num_gpu=args.num_gpu,
#         main_gpu=args.main_gpu,
#     )

#     evaluated_df = evaluate_dataset_rows(
#         df=df,
#         model=args.model,
#         url=args.url,
#         timeout=args.timeout,
#         num_gpu=args.num_gpu,
#         main_gpu=args.main_gpu,
#         max_rows=args.max_rows,
#     )

#     evaluated_df.to_csv(args.output, index=False)
#     print(f"[STEP 13] Saved {len(evaluated_df)} evaluated path rows to: {args.output}")


# if __name__ == "__main__":
#     main()

"""
Step 13: LLM-based propagation path evaluation

Pipeline:
1. Load candidate propagation paths directly from a CSV, or derive them from Step 12 outputs.
2. Build a prompt for each anomaly/path pair.
3. Call a local Ollama model to judge path plausibility.
4. Parse the JSON response.
5. Save row-level path evaluation results to CSV.
"""

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import requests


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

STEP12_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "step12")
STEP13_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "step13")
os.makedirs(STEP13_DIR, exist_ok=True)

DEFAULT_INPUT_PATH = os.path.join(STEP12_DIR, "swat_rca_step12_results.csv")
DEFAULT_OUTPUT_PATH = os.path.join(STEP13_DIR, "all_propagation_path_llm_evaluation.csv")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_TIMEOUT = int(os.getenv("STEP13_OLLAMA_TIMEOUT", "120"))
STEP13_PROGRESS_EVERY = int(os.getenv("STEP13_PROGRESS_EVERY", "25"))
STEP13_MAX_RETRIES = int(os.getenv("STEP13_MAX_RETRIES", "3"))
STEP13_RETRY_DELAY = float(os.getenv("STEP13_RETRY_DELAY", "2"))
STEP13_OLLAMA_NUM_GPU = int(os.getenv("STEP13_OLLAMA_NUM_GPU", "-1"))
STEP13_OLLAMA_MAIN_GPU = os.getenv("STEP13_OLLAMA_MAIN_GPU")
STEP13_OLLAMA_NUM_BATCH = int(os.getenv("STEP13_OLLAMA_NUM_BATCH", "512"))
STEP13_OLLAMA_NUM_CTX = int(os.getenv("STEP13_OLLAMA_NUM_CTX", "4096"))
STEP13_MAX_WORKERS = int(os.getenv("STEP13_MAX_WORKERS", "3"))
STEP13_LOG_MODE = os.getenv("STEP13_LOG_MODE", "errors").strip().lower()
STEP13_FLUSH_EVERY = int(os.getenv("STEP13_FLUSH_EVERY", "0"))

if STEP13_OLLAMA_MAIN_GPU is not None and STEP13_OLLAMA_MAIN_GPU != "":
    STEP13_OLLAMA_MAIN_GPU = int(STEP13_OLLAMA_MAIN_GPU)
else:
    STEP13_OLLAMA_MAIN_GPU = None

if STEP13_LOG_MODE not in {"off", "errors", "all"}:
    STEP13_LOG_MODE = "errors"

SWAT_DATASET_CONTEXT = (
    "SWaT stands for Secure Water Treatment. It is a cyber-physical industrial water treatment "
    "dataset with time-series measurements from sensors and actuators across six process stages. "
    "In this repository, anomaly detection is performed on continuous process-value sensors ending in .Pv, "
    "and candidate propagation paths come from Bayesian Network root-cause analysis."
)

STAGE_CONTEXT = {
    "P1": {
        "name": "Raw Water Intake",
        "description": "Intake-side stage focused on incoming water level and inlet flow behavior.",
        "modeled_variables": ["LIT101.Pv", "FIT101.Pv"],
    },
    "P2": {
        "name": "Pre-treatment",
        "description": "Pre-treatment stage with one flow signal and multiple analyzer measurements.",
        "modeled_variables": ["FIT201.Pv", "AIT201.Pv", "AIT202.Pv", "AIT203.Pv"],
    },
    "P3": {
        "name": "Ultra-Filtration",
        "description": "Ultra-filtration stage combining analyzer, level, flow, and differential pressure signals.",
        "modeled_variables": ["AIT301.Pv", "AIT302.Pv", "AIT303.Pv", "LIT301.Pv", "FIT301.Pv", "DPIT301.Pv"],
    },
    "P4": {
        "name": "De-Chlorination",
        "description": "De-chlorination stage with analyzer, level, flow, and pressure measurements.",
        "modeled_variables": ["AIT401.Pv", "AIT402.Pv", "LIT401.Pv", "FIT401.Pv"],
    },
    "P5": {
        "name": "Reverse Osmosis",
        "description": "Reverse-osmosis stage with analyzer, flow, and pressure behavior across membrane-related equipment.",
        "modeled_variables": [
            "AIT501.Pv",
            "AIT502.Pv",
            "AIT503.Pv",
            "AIT504.Pv",
            "FIT501.Pv",
            "FIT502.Pv",
            "FIT503.Pv",
            "FIT504.Pv",
            "PIT501.Pv",
            "PIT502.Pv",
            "PIT503.Pv",
        ],
    },
    "P6": {
        "name": "Disposition",
        "description": "Final disposition stage with a very small continuous measurement set in this repository.",
        "modeled_variables": ["FIT601.Pv"],
    },
}

SENSOR_PREFIX_CONTEXT = {
    "FIT": "flow indicator transmitter, usually a flow-related measurement",
    "LIT": "level indicator transmitter, usually a tank or vessel level measurement",
    "AIT": "analyzer indicator transmitter, usually a water quality or chemistry measurement",
    "PIT": "pressure indicator transmitter, usually a pressure measurement",
    "DPIT": "differential pressure indicator transmitter, usually a pressure-drop or filtration-resistance measurement",
}


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


def parse_sensor_list(value: Any) -> List[str]:
    parsed = _load_structured_value(value)
    if parsed is None:
        return []

    if isinstance(parsed, (list, tuple, set)):
        return [str(item).strip() for item in parsed if str(item).strip()]

    if isinstance(parsed, str):
        if "->" in parsed:
            return [part.strip() for part in parsed.split("->") if part.strip()]
        if "," in parsed:
            return [part.strip() for part in parsed.split(",") if part.strip()]
        return [parsed.strip()] if parsed.strip() else []

    return [str(parsed).strip()]


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


def format_path(path_nodes: List[str]) -> str:
    return " -> ".join(path_nodes)


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


def normalize_input_rows(df: pd.DataFrame) -> pd.DataFrame:
    if {"anomaly_sensors", "candidate_path"}.issubset(df.columns):
        normalized = df.copy()
        normalized["anomaly_sensors"] = normalized["anomaly_sensors"].apply(parse_sensor_list)
        normalized["candidate_path_nodes"] = normalized["candidate_path"].apply(
            lambda value: parse_candidate_paths(value)[0] if parse_candidate_paths(value) else []
        )
        normalized["candidate_path"] = normalized["candidate_path_nodes"].apply(format_path)
        return normalized

    if {"anomalous_sensors", "propagation_paths"}.issubset(df.columns):
        expanded_rows: List[Dict[str, Any]] = []

        for _, row in df.iterrows():
            row_dict = row.to_dict()
            anomaly_sensors = parse_sensor_list(row_dict.get("anomalous_sensors"))
            candidate_paths = parse_candidate_paths(row_dict.get("propagation_paths"))

            for path_index, path_nodes in enumerate(candidate_paths, start=1):
                expanded_row = dict(row_dict)
                expanded_row["anomaly_sensors"] = anomaly_sensors
                expanded_row["candidate_path_index"] = path_index
                expanded_row["candidate_path_nodes"] = path_nodes
                expanded_row["candidate_path"] = format_path(path_nodes)
                expanded_rows.append(expanded_row)

        return pd.DataFrame(expanded_rows)

    raise ValueError(
        "Input CSV must contain either 'anomaly_sensors' and 'candidate_path', "
        "or Step 12 columns 'anomalous_sensors' and 'propagation_paths'."
    )


def build_prompt(row: Dict[str, Any]) -> str:
    anomaly_id = row.get("anomaly_id", "unknown")
    stage = row.get("stage", "unknown")
    phase = row.get("phase", "unknown")
    timestamp = row.get("timestamp", "unknown")
    sensors = row.get("anomaly_sensors", [])
    path_nodes = row.get("candidate_path_nodes", [])
    stage_context = build_stage_context(stage)
    sensor_context = build_sensor_context(sensors + path_nodes)

    payload = {
        "anomaly_id": anomaly_id,
        "stage": stage,
        "stage_name": stage_context["name"],
        "phase": phase,
        "timestamp": timestamp,
        "anomaly_sensors": sensors,
        "candidate_path": path_nodes,
        "stage_modeled_variables": stage_context["modeled_variables"],
    }

    return (
        "You are evaluating anomaly propagation paths for the SWaT industrial control system. "
        "Assess whether the candidate path is a plausible explanation for the observed anomaly.\n\n"
        f"Dataset context: {SWAT_DATASET_CONTEXT}\n"
        f"Stage context: {stage} is {stage_context['name']}. {stage_context['description']}\n"
        "The stage-wise model in this repository uses only continuous .Pv process values, so the anomaly sensors and path nodes are continuous measurements rather than actuator states or alarm flags.\n"
        f"Modeled variables for this stage: {', '.join(stage_context['modeled_variables']) if stage_context['modeled_variables'] else 'unknown'}\n"
        f"Sensor tag meanings in this case: {'; '.join(sensor_context) if sensor_context else 'no additional tag context available'}\n\n"
        "Evaluate exactly these criteria:\n"
        "1. Sensor coverage: does the path cover the anomalous sensors, fully or partially?\n"
        "2. Causal direction plausibility: does the path order look like a realistic propagation direction?\n"
        "3. Propagation completeness: does the path form a coherent spread instead of skipping key sensors?\n\n"
        "Guidance:\n"
        "- Be conservative. Mark invalid if the path clearly fails to explain the anomaly sensors.\n"
        "- A path can still be valid if it only partially covers the anomaly sensors but remains causally plausible.\n"
        "- Prefer explanations that stay within the same stage and align with the measurement types involved.\n"
        "- Treat flow, level, analyzer, pressure, and differential pressure signals as different but related physical measurements; use that to judge whether the path order looks coherent.\n"
        "- Use the phase label only as supporting context, not as the main reason to accept or reject a path.\n"
        "- Confidence must be a number between 0 and 1.\n"
        "- Reason must be short and specific.\n\n"
        "Return only valid JSON with this exact schema:\n"
        "{\n"
        '  "valid": true,\n'
        '  "confidence": 0.0,\n'
        '  "reason": "short explanation"\n'
        "}\n\n"
        f"Case data:\n{json.dumps(payload, ensure_ascii=True)}"
    )


def extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError(f"No JSON object found in model response: {text}")
        return json.loads(match.group(0))


def normalize_llm_result(result: Dict[str, Any]) -> Dict[str, Any]:
    valid = result.get("valid", False)
    if isinstance(valid, str):
        valid = valid.strip().lower() == "true"
    else:
        valid = bool(valid)

    confidence = result.get("confidence", 0.0)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    reason = str(result.get("reason", "No explanation returned.")).strip()
    if not reason:
        reason = "No explanation returned."

    return {
        "valid": valid,
        "confidence": confidence,
        "reason": reason,
    }


def call_ollama(
    prompt: str,
    session: requests.Session,
    model: str = OLLAMA_MODEL,
    url: str = OLLAMA_URL,
    timeout: int = OLLAMA_TIMEOUT,
    num_gpu: int = STEP13_OLLAMA_NUM_GPU,
    main_gpu: Optional[int] = STEP13_OLLAMA_MAIN_GPU,
    num_batch: int = STEP13_OLLAMA_NUM_BATCH,
    num_ctx: int = STEP13_OLLAMA_NUM_CTX,
    log_mode: str = STEP13_LOG_MODE,
    max_retries: int = STEP13_MAX_RETRIES,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1,
            "num_gpu": num_gpu,
            "num_batch": num_batch,
            "num_ctx": num_ctx,
        },
    }

    if main_gpu is not None:
        payload["options"]["main_gpu"] = main_gpu

    last_error: Optional[Exception] = None

    log_path = os.path.join(STEP13_DIR, "step13_llm_debug.log")
    effective_log_mode = (log_mode or "errors").strip().lower()
    if effective_log_mode not in {"off", "errors", "all"}:
        effective_log_mode = "errors"

    should_log_all = effective_log_mode == "all"
    should_log_errors = effective_log_mode in {"errors", "all"}

    for attempt in range(1, max_retries + 1):
        call_start = time.time()
        try:
            if should_log_all:
                with open(log_path, "a", encoding="utf-8") as logf:
                    logf.write(f"\n---\n[CALL] Attempt {attempt} at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    logf.write(f"Prompt:\n{prompt}\n")

            response = session.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            body = response.json()
            model_text = body.get("response", "")
            call_end = time.time()

            if should_log_all:
                with open(log_path, "a", encoding="utf-8") as logf:
                    logf.write(f"Response:\n{model_text}\n")
                    logf.write(f"Call duration: {call_end - call_start:.2f} seconds\n")

            if not model_text:
                raise ValueError(f"Ollama response is missing 'response': {body}")
            return normalize_llm_result(extract_json_object(model_text))
        except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
            call_end = time.time()
            last_error = exc

            if should_log_errors:
                with open(log_path, "a", encoding="utf-8") as logf:
                    logf.write(f"\n---\n[ERROR] Attempt {attempt} at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    logf.write(f"Error: {repr(exc)}\n")
                    logf.write(f"Call duration: {call_end - call_start:.2f} seconds\n")

            if attempt == max_retries:
                break
            time.sleep(STEP13_RETRY_DELAY * attempt)

    raise RuntimeError(f"Failed to evaluate path with Ollama after {max_retries} attempts: {last_error}")


def append_rows_to_csv(rows: List[Dict[str, Any]], output_path: str) -> int:
    if not rows:
        return 0

    write_header = not os.path.exists(output_path) or os.path.getsize(output_path) == 0
    batch_df = pd.DataFrame(rows)
    batch_df.to_csv(output_path, mode="a", index=False, header=write_header)
    return len(rows)


def evaluate_dataset_rows(
    df: pd.DataFrame,
    model: str = OLLAMA_MODEL,
    url: str = OLLAMA_URL,
    timeout: int = OLLAMA_TIMEOUT,
    num_gpu: int = STEP13_OLLAMA_NUM_GPU,
    main_gpu: Optional[int] = STEP13_OLLAMA_MAIN_GPU,
    num_batch: int = STEP13_OLLAMA_NUM_BATCH,
    num_ctx: int = STEP13_OLLAMA_NUM_CTX,
    max_workers: int = STEP13_MAX_WORKERS,
    log_mode: str = STEP13_LOG_MODE,
    output_path: Optional[str] = None,
    flush_every: int = STEP13_FLUSH_EVERY,
    return_dataframe: bool = True,
    max_rows: Optional[int] = None,
) -> Any:
    normalized_df = normalize_input_rows(df)
    if normalized_df.empty:
        return normalized_df

    if max_rows is not None:
        normalized_df = normalized_df.head(max_rows).copy()

    source_rows = normalized_df.to_dict(orient="records")
    total_rows = len(source_rows)
    worker_count = max(1, min(max_workers, total_rows))
    print(f"[STEP 13] Using {worker_count} parallel worker(s)")

    def evaluate_one_row(row: Dict[str, Any]) -> Dict[str, Any]:
        with requests.Session() as session:
            try:
                evaluation = call_ollama(
                    prompt=build_prompt(row),
                    session=session,
                    model=model,
                    url=url,
                    timeout=timeout,
                    num_gpu=num_gpu,
                    main_gpu=main_gpu,
                    num_batch=num_batch,
                    num_ctx=num_ctx,
                    log_mode=log_mode,
                )
            except Exception as exc:
                evaluation = {
                    "valid": False,
                    "confidence": 0.0,
                    "reason": f"Evaluation failed: {exc}",
                }

        result_row = dict(row)
        result_row["path_valid"] = evaluation["valid"]
        result_row["confidence"] = evaluation["confidence"]
        result_row["reason"] = evaluation["reason"]

        if isinstance(result_row.get("anomaly_sensors"), list):
            result_row["anomaly_sensors"] = json.dumps(result_row["anomaly_sensors"])
        if isinstance(result_row.get("candidate_path_nodes"), list):
            result_row["candidate_path_nodes"] = json.dumps(result_row["candidate_path_nodes"])

        return result_row

    completed = 0
    flushed = 0
    ordered_results: List[Optional[Dict[str, Any]]] = [None] * total_rows if return_dataframe else []
    in_order_cache: Dict[int, Dict[str, Any]] = {}
    next_to_flush = 0
    flush_buffer: List[Dict[str, Any]] = []

    flush_interval = max(0, flush_every)
    flush_enabled = bool(output_path and flush_interval > 0)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(evaluate_one_row, row): index
            for index, row in enumerate(source_rows)
        }

        for future in as_completed(futures):
            index = futures[future]
            result_row = future.result()

            if return_dataframe:
                ordered_results[index] = result_row

            if flush_enabled:
                in_order_cache[index] = result_row
                while next_to_flush in in_order_cache:
                    flush_buffer.append(in_order_cache.pop(next_to_flush))
                    next_to_flush += 1

                if len(flush_buffer) >= flush_interval:
                    flushed += append_rows_to_csv(flush_buffer, output_path)
                    flush_buffer = []

            completed += 1

            if completed % max(1, STEP13_PROGRESS_EVERY) == 0 or completed == total_rows:
                if flush_enabled:
                    print(
                        f"[STEP 13] Progress: {completed}/{total_rows} paths evaluated "
                        f"({flushed} rows checkpointed)"
                    )
                else:
                    print(f"[STEP 13] Progress: {completed}/{total_rows} paths evaluated")

    if flush_enabled and flush_buffer:
        flushed += append_rows_to_csv(flush_buffer, output_path)
        print(f"[STEP 13] Final checkpoint flush complete: {flushed} rows written")

    if return_dataframe:
        final_results = [row for row in ordered_results if row is not None]
        return pd.DataFrame(final_results)

    return completed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate anomaly propagation paths with a local Ollama LLM.")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="Input CSV path.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Output CSV path.")
    parser.add_argument("--model", default=OLLAMA_MODEL, help="Ollama model name.")
    parser.add_argument("--url", default=OLLAMA_URL, help="Ollama generate API URL.")
    parser.add_argument("--timeout", type=int, default=OLLAMA_TIMEOUT, help="HTTP timeout in seconds.")
    parser.add_argument(
        "--num-gpu",
        type=int,
        default=STEP13_OLLAMA_NUM_GPU,
        help="Ollama num_gpu option. Use 0 to force CPU, positive values to offload layers, -1 for Ollama default.",
    )
    parser.add_argument(
        "--main-gpu",
        type=int,
        default=STEP13_OLLAMA_MAIN_GPU,
        help="Optional Ollama main_gpu index (for multi-GPU systems).",
    )
    parser.add_argument("--num-batch", type=int, default=STEP13_OLLAMA_NUM_BATCH, help="Ollama num_batch option.")
    parser.add_argument("--num-ctx", type=int, default=STEP13_OLLAMA_NUM_CTX, help="Ollama num_ctx option.")
    parser.add_argument("--max-workers", type=int, default=STEP13_MAX_WORKERS, help="Parallel worker count for row evaluation.")
    parser.add_argument(
        "--flush-every",
        type=int,
        default=STEP13_FLUSH_EVERY,
        help="Checkpoint rows to CSV every N in-order results (0 disables incremental flush).",
    )
    parser.add_argument(
        "--log-mode",
        choices=["off", "errors", "all"],
        default=STEP13_LOG_MODE,
        help="Logging mode for per-call debug logs: off, errors, or all.",
    )
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row limit for quick runs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Start timing
    start_time = time.time()

    print(f"[STEP 13] Loading input CSV: {args.input}")
    df = pd.read_csv(args.input)
    print(f"[STEP 13] Loaded {len(df)} source rows")
    print(
        "[STEP 13] Ollama options: "
        f"model={args.model}, num_gpu={args.num_gpu}, "
        f"main_gpu={args.main_gpu if args.main_gpu is not None else 'auto'}, "
        f"num_batch={args.num_batch}, num_ctx={args.num_ctx}, "
        f"workers={max(1, args.max_workers)}, flush_every={max(0, args.flush_every)}, "
        f"log_mode={args.log_mode}"
    )

    # If output is the default, append a timestamp to avoid overwriting
    output_path = args.output
    if args.output == DEFAULT_OUTPUT_PATH:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(DEFAULT_OUTPUT_PATH)
        output_path = f"{base}_{timestamp}{ext}"

    if max(0, args.flush_every) > 0:
        if os.path.exists(output_path):
            os.remove(output_path)

        evaluated_count = evaluate_dataset_rows(
            df=df,
            model=args.model,
            url=args.url,
            timeout=args.timeout,
            num_gpu=args.num_gpu,
            main_gpu=args.main_gpu,
            num_batch=args.num_batch,
            num_ctx=args.num_ctx,
            max_workers=max(1, args.max_workers),
            log_mode=args.log_mode,
            output_path=output_path,
            flush_every=max(0, args.flush_every),
            return_dataframe=False,
            max_rows=args.max_rows,
        )
        print(f"[STEP 13] Saved {evaluated_count} evaluated path rows to: {output_path}")
    else:
        evaluated_df = evaluate_dataset_rows(
            df=df,
            model=args.model,
            url=args.url,
            timeout=args.timeout,
            num_gpu=args.num_gpu,
            main_gpu=args.main_gpu,
            num_batch=args.num_batch,
            num_ctx=args.num_ctx,
            max_workers=max(1, args.max_workers),
            log_mode=args.log_mode,
            max_rows=args.max_rows,
        )
        evaluated_df.to_csv(output_path, index=False)
        print(f"[STEP 13] Saved {len(evaluated_df)} evaluated path rows to: {output_path}")

    # End timing
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"[STEP 13] Total elapsed time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")


if __name__ == "__main__":
    main()