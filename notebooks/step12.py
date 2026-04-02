"""
Step 12: Root Cause Analysis using Bayesian Networks

Pipeline:
1. Load anomaly windows (from Step 8)
2. For each anomaly:
    - Identify stage
    - Load BN
    - Perform inference
    - Rank root causes
    - Extract propagation paths
    - Compute confidence score
3. Store RCA result in Weaviate
"""

import os
import json
import uuid
import numpy as np
import pandas as pd
from collections import defaultdict
from urllib.parse import urlparse
from sklearn.preprocessing import KBinsDiscretizer

try:
    from pgmpy.models import DiscreteBayesianNetwork as PgmpyBayesianModel
except ImportError:
    from pgmpy.models import BayesianNetwork as PgmpyBayesianModel
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import MaximumLikelihoodEstimator

import weaviate

# ---------------- CONFIG ---------------- #
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

STEP8_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "step8")
STEP6_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "step6")
STEP11_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "step11")
STEP12_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "step12")
os.makedirs(STEP12_DIR, exist_ok=True)

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_BEACON_HOST = urlparse(WEAVIATE_URL).hostname or "localhost"

TOP_K_ROOTS = 3
STEP12_PROGRESS_EVERY = int(os.getenv("STEP12_PROGRESS_EVERY", "200"))
STEP12_VERBOSE_PER_ANOMALY = os.getenv("STEP12_VERBOSE_PER_ANOMALY", "0") == "1"
STEP12_RESET_WEAVIATE = os.getenv("STEP12_RESET_WEAVIATE", "1") == "1"


# ---------------- WEAVIATE ---------------- #

def get_weaviate_client():
    client = weaviate.Client(WEAVIATE_URL)
    if not client.is_ready():
        raise RuntimeError("Weaviate not ready")
    return client


def ensure_rcaresult_class(client):
    schema = {
        "class": "RCAResult",
        "properties": [
            {"name": "anomaly_id", "dataType": ["string"]},
            {"name": "stage", "dataType": ["string"]},
            {"name": "timestamp", "dataType": ["string"]},
            {"name": "phase", "dataType": ["string"]},
            {"name": "guilty_feature", "dataType": ["string"]},
            {"name": "guilty_feature_score", "dataType": ["number"]},
            {"name": "anomalous_sensors", "dataType": ["string"]},
            {"name": "anomalous_sensor_scores", "dataType": ["string"]},
            {"name": "root_causes", "dataType": ["string"]},
            {"name": "propagation_paths", "dataType": ["string"]},
            {"name": "confidence", "dataType": ["number"]},
        ],
    }

    current_schema = client.schema.get()
    classes = current_schema.get("classes", [])
    existing_classes = {c.get("class") for c in classes}
    if "RCAResult" not in existing_classes:
        client.schema.create_class(schema)
        return

    class_schema = next((c for c in classes if c.get("class") == "RCAResult"), None) or {}
    existing_properties = {p.get("name") for p in class_schema.get("properties", [])}
    for prop in schema["properties"]:
        if prop["name"] not in existing_properties:
            client.schema.property.create("RCAResult", prop)


def reset_rcaresult_class(client):
    current_schema = client.schema.get()
    class_names = {c.get("class") for c in current_schema.get("classes", [])}
    if "RCAResult" in class_names:
        client.schema.delete_class("RCAResult")
    ensure_rcaresult_class(client)


def make_rca_uuid(anomaly_id, stage, timestamp):
    key = f"RCAResult|{anomaly_id}|{stage}|{'' if timestamp is None else timestamp}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def store_rca_result(
    client,
    anomaly_id,
    stage,
    timestamp,
    phase,
    guilty_feature,
    guilty_feature_score,
    anomalous_sensors,
    anomalous_sensor_scores,
    root_causes,
    propagation_paths,
    confidence,
):
    object_uuid = make_rca_uuid(anomaly_id, stage, timestamp)

    data_object = {
        "anomaly_id": anomaly_id,
        "stage": stage,
        "timestamp": "" if timestamp is None else str(timestamp),
        "phase": "" if phase is None else str(phase),
        "guilty_feature": "" if guilty_feature is None else str(guilty_feature),
        "guilty_feature_score": None if guilty_feature_score is None else float(guilty_feature_score),
        "anomalous_sensors": json.dumps(anomalous_sensors),
        "anomalous_sensor_scores": json.dumps(anomalous_sensor_scores),
        "root_causes": json.dumps(root_causes),
        "propagation_paths": json.dumps(propagation_paths),
        "confidence": confidence
    }

    try:
        client.data_object.replace(
            uuid=object_uuid,
            data_object=data_object,
            class_name="RCAResult",
        )
    except Exception:
        client.data_object.create(
            uuid=object_uuid,
            data_object=data_object,
            class_name="RCAResult",
        )


def to_float_dict(values):
    result = {}
    for key, value in values.items():
        try:
            result[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return result


def save_step12_results_csv(rows):
    output_path = os.path.join(STEP12_DIR, "swat_rca_step12_results.csv")
    df = pd.DataFrame(rows)
    if not df.empty:
        for column in [
            "anomalous_sensors",
            "anomalous_sensor_scores",
            "root_causes",
            "propagation_paths",
        ]:
            if column in df.columns:
                df[column] = df[column].apply(json.dumps)

    df.to_csv(output_path, index=False)
    return output_path, len(df)


# ---------------- BN LOADING ---------------- #

def load_bn_from_json(stage):
    path = os.path.join(STEP11_DIR, f"BN_{stage}.json")
    with open(path) as f:
        bn_dict = json.load(f)

    model = PgmpyBayesianModel([(e["from"], e["to"]) for e in bn_dict["edges"]])
    model.add_nodes_from(bn_dict.get("nodes", []))

    cpts = bn_dict.get("cpts", {})
    has_any_cpt = isinstance(cpts, dict) and any(cpt_data is not None for cpt_data in cpts.values())

    if not has_any_cpt:
        return fit_bn_parameters_from_stage_data(model, stage)

    # Rebuild CPTs
    for node, cpt_data in cpts.items():
        if cpt_data is None:
            continue

        if node not in model.nodes():
            continue

        if isinstance(cpt_data, dict) and "values" in cpt_data:
            values = cpt_data["values"]
            variable_card = int(cpt_data["variable_card"])
            evidence = cpt_data.get("evidence") or None
            evidence_card = cpt_data.get("evidence_card") or None

            cpd = TabularCPD(
                variable=node,
                variable_card=variable_card,
                values=values,
                evidence=evidence,
                evidence_card=evidence_card,
            )

            model.add_cpds(cpd)
            continue

        df = pd.DataFrame(cpt_data)
        values = df.values.T
        evidence = list(model.get_parents(node))
        evidence_card = [len(df.columns)] * len(evidence)

        cpd = TabularCPD(
            variable=node,
            variable_card=len(df),
            values=values,
            evidence=evidence if evidence else None,
            evidence_card=evidence_card if evidence else None,
        )

        model.add_cpds(cpd)

    try:
        model.check_model()
        return model
    except Exception:
        return fit_bn_parameters_from_stage_data(model, stage)


def fit_bn_parameters_from_stage_data(model, stage):
    stage_data_path = os.path.join(STEP6_DIR, f"X_train_{stage}.npy")
    feature_map_path = os.path.join(STEP6_DIR, "stage_feature_map.json")

    if not os.path.exists(stage_data_path) or not os.path.exists(feature_map_path):
        raise FileNotFoundError(f"Missing stage data to fit BN CPTs for {stage}")

    X_train = np.load(stage_data_path)
    with open(feature_map_path) as f:
        stage_feature_map = json.load(f)

    feature_names = stage_feature_map.get(stage, [])
    if len(feature_names) == 0:
        raise ValueError(f"No features found for stage {stage}")

    df_raw = pd.DataFrame(X_train.mean(axis=1), columns=feature_names)
    df_filtered = df_raw.loc[:, df_raw.std() > 1e-6]
    df = df_filtered if not df_filtered.empty else df_raw

    model_nodes = [node for node in model.nodes() if node in df.columns]
    if not model_nodes:
        raise ValueError(f"No model nodes overlap available stage data for {stage}")

    sub_edges = [(u, v) for (u, v) in model.edges() if u in model_nodes and v in model_nodes]
    fitted_model = PgmpyBayesianModel(sub_edges)
    fitted_model.add_nodes_from(model_nodes)

    kbd = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    df_discrete = pd.DataFrame(kbd.fit_transform(df[model_nodes]), columns=model_nodes).astype(int)
    fitted_model.fit(df_discrete, estimator=MaximumLikelihoodEstimator)
    fitted_model.check_model()
    return fitted_model


def load_anomalies():
    anomaly_file = os.path.join(STEP8_DIR, "anomalies.json")
    if os.path.exists(anomaly_file):
        with open(anomaly_file) as f:
            anomalies = json.load(f)
        return anomalies if isinstance(anomalies, list) else []

    fallback_csv = os.path.join(STEP8_DIR, "swat_rca_results.csv")
    if os.path.exists(fallback_csv):
        try:
            df = pd.read_csv(fallback_csv)
        except pd.errors.EmptyDataError:
            return []

        if df.empty:
            return []

        anomalies = []
        for idx, row in df.iterrows():
            stage = row.get("guilty_stage")
            if pd.isna(stage):
                continue

            guilty_feature = row.get("guilty_feature")
            guilty_feature_score = row.get("feature_score")
            deviating_sensors = []
            if pd.notna(guilty_feature) and str(guilty_feature).strip():
                deviating_sensors = [str(guilty_feature)]

            anomalous_sensors = deviating_sensors
            anomalous_sensor_scores = {}

            anomalous_sensors_raw = row.get("anomalous_sensors")
            if pd.notna(anomalous_sensors_raw):
                if isinstance(anomalous_sensors_raw, str):
                    try:
                        parsed = json.loads(anomalous_sensors_raw)
                        if isinstance(parsed, list):
                            anomalous_sensors = [str(sensor) for sensor in parsed]
                    except json.JSONDecodeError:
                        pass
                elif isinstance(anomalous_sensors_raw, list):
                    anomalous_sensors = [str(sensor) for sensor in anomalous_sensors_raw]

            anomalous_scores_raw = row.get("anomalous_sensor_scores")
            if pd.notna(anomalous_scores_raw):
                if isinstance(anomalous_scores_raw, str):
                    try:
                        parsed_scores = json.loads(anomalous_scores_raw)
                        if isinstance(parsed_scores, dict):
                            anomalous_sensor_scores = {
                                str(sensor): float(score)
                                for sensor, score in parsed_scores.items()
                            }
                    except (json.JSONDecodeError, ValueError, TypeError):
                        pass
                elif isinstance(anomalous_scores_raw, dict):
                    anomalous_sensor_scores = {
                        str(sensor): float(score)
                        for sensor, score in anomalous_scores_raw.items()
                    }

            if not anomalous_sensor_scores and pd.notna(guilty_feature_score) and deviating_sensors:
                anomalous_sensor_scores = {deviating_sensors[0]: float(guilty_feature_score)}

            anomalies.append(
                {
                    "id": f"csv_{idx}",
                    "stage": str(stage),
                    "deviating_sensors": deviating_sensors,
                    "evidence_bins": {},
                    "anomalous_sensors": anomalous_sensors,
                    "anomalous_sensor_scores": anomalous_sensor_scores,
                    "guilty_feature": None if pd.isna(guilty_feature) else str(guilty_feature),
                    "guilty_feature_score": None if pd.isna(guilty_feature_score) else float(guilty_feature_score),
                    "timestamp": None if pd.isna(row.get("t_stamp")) else str(row.get("t_stamp")),
                    "phase": None if pd.isna(row.get("phase")) else str(row.get("phase")),
                }
            )

        return anomalies

    return []


# ---------------- ROOT CAUSE LOGIC ---------------- #

def compute_root_scores(model, inference, evidence):
    scores = {}

    for node in model.nodes():
        if node in evidence:
            continue

        posterior = inference.query([node], evidence=evidence, show_progress=False)

        if hasattr(posterior, "values"):
            values = np.asarray(posterior.values).reshape(-1)
        else:
            factor = posterior[node]
            values = np.asarray(factor.values).reshape(-1)

        if values.size == 0:
            prob_abnormal = 0.0
        else:
            prob_abnormal = float(values[-1])  # highest bin = abnormal

        scores[node] = prob_abnormal

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def extract_propagation_paths(model, root, deviating_nodes):
    paths = []

    def dfs(current, path):
        if current in deviating_nodes:
            paths.append(path.copy())
        for child in model.get_children(current):
            dfs(child, path + [child])

    dfs(root, [root])
    return paths


def compute_confidence(root_scores, propagation_paths):
    if not root_scores:
        return 0.0

    avg_root_prob = np.mean([score for _, score in root_scores])
    avg_path_length = np.mean([len(p) for p in propagation_paths]) if propagation_paths else 1

    confidence = float(avg_root_prob * (1 / avg_path_length))
    return min(confidence, 1.0)


# ---------------- MAIN RCA PIPELINE ---------------- #

def run_stage12_rca():

    anomalies = load_anomalies()
    if not anomalies:
        print("[STEP 12] No anomalies found in anomalies.json or swat_rca_results.csv. Nothing to process.")
        return

    client = get_weaviate_client()
    if STEP12_RESET_WEAVIATE:
        print("[STEP 12] Resetting RCAResult class for a clean, duplicate-free run...")
        reset_rcaresult_class(client)
    else:
        ensure_rcaresult_class(client)

    total_anomalies = len(anomalies)
    print(f"[STEP 12] Starting RCA for {total_anomalies} anomalies...")

    saved_rows = []

    for index, anomaly in enumerate(anomalies, start=1):
        anomaly_id = anomaly.get("id", "unknown")
        stage = anomaly.get("stage")
        timestamp = anomaly.get("timestamp")
        phase = anomaly.get("phase")
        deviating_nodes = anomaly.get("deviating_sensors", [])
        if isinstance(deviating_nodes, str):
            deviating_nodes = [deviating_nodes]

        anomalous_sensors = anomaly.get("anomalous_sensors", deviating_nodes)
        if isinstance(anomalous_sensors, str):
            try:
                parsed = json.loads(anomalous_sensors)
                anomalous_sensors = parsed if isinstance(parsed, list) else [anomalous_sensors]
            except json.JSONDecodeError:
                anomalous_sensors = [anomalous_sensors]

        anomalous_sensor_scores = anomaly.get("anomalous_sensor_scores", {})
        if isinstance(anomalous_sensor_scores, str):
            try:
                parsed_scores = json.loads(anomalous_sensor_scores)
                anomalous_sensor_scores = parsed_scores if isinstance(parsed_scores, dict) else {}
            except json.JSONDecodeError:
                anomalous_sensor_scores = {}

        if not isinstance(anomalous_sensor_scores, dict):
            anomalous_sensor_scores = {}
        anomalous_sensor_scores = to_float_dict(anomalous_sensor_scores)

        guilty_feature = anomaly.get("guilty_feature")
        guilty_feature_score = anomaly.get("guilty_feature_score")
        if guilty_feature_score is not None:
            try:
                guilty_feature_score = float(guilty_feature_score)
            except (ValueError, TypeError):
                guilty_feature_score = None

        if not anomalous_sensors and guilty_feature:
            anomalous_sensors = [str(guilty_feature)]

        if not anomalous_sensor_scores and guilty_feature and guilty_feature_score is not None:
            anomalous_sensor_scores = {str(guilty_feature): float(guilty_feature_score)}

        if not stage:
            print(f"[STEP 12] Skipping anomaly {anomaly_id}: missing stage")
            continue

        model = load_bn_from_json(stage)
        inference = VariableElimination(model)

        evidence_raw = anomaly.get("evidence_bins", {}) or {}
        evidence = {k: v for k, v in evidence_raw.items() if k in model.nodes()}

        root_scores = compute_root_scores(model, inference, evidence)
        top_roots = root_scores[:TOP_K_ROOTS]

        propagation_paths = []
        for root, _ in top_roots:
            propagation_paths.extend(
                extract_propagation_paths(model, root, deviating_nodes)
            )

        confidence = compute_confidence(top_roots, propagation_paths)

        store_rca_result(
            client,
            anomaly_id,
            stage,
            timestamp,
            phase,
            guilty_feature,
            guilty_feature_score,
            anomalous_sensors,
            anomalous_sensor_scores,
            top_roots,
            propagation_paths,
            confidence
        )

        saved_rows.append(
            {
                "anomaly_id": anomaly_id,
                "stage": stage,
                "timestamp": timestamp,
                "phase": phase,
                "guilty_feature": guilty_feature,
                "guilty_feature_score": guilty_feature_score,
                "anomalous_sensors": anomalous_sensors,
                "anomalous_sensor_scores": anomalous_sensor_scores,
                "root_causes": top_roots,
                "propagation_paths": propagation_paths,
                "confidence": confidence,
            }
        )

        if STEP12_VERBOSE_PER_ANOMALY:
            top_root = top_roots[0][0] if top_roots else "none"
            print(
                f"[STEP 12] Done {index}/{total_anomalies} | id={anomaly_id} | "
                f"stage={stage} | top_root={top_root} | conf={confidence:.4f}"
            )
        elif index % max(1, STEP12_PROGRESS_EVERY) == 0 or index == total_anomalies:
            print(f"[STEP 12] Progress: {index}/{total_anomalies} anomalies processed")

    csv_path, csv_rows = save_step12_results_csv(saved_rows)
    unique_timestamps = len({row.get("timestamp") for row in saved_rows if row.get("timestamp")})

    print("\n[STEP 12 COMPLETED] RCA stored in Weaviate.")
    print(f"[STEP 12] CSV saved to: {csv_path}")
    print(f"[STEP 12] Saved rows: {csv_rows} | Unique timestamps: {unique_timestamps}")


if __name__ == "__main__":
    run_stage12_rca()

