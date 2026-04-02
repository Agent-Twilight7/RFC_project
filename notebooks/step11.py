"""
Step 11: Learning Causal Graphs (Bayesian Networks)
1. Loads stage-wise normal data (from Step 6)
2. Aggregates time-series windows
3. Optionally discretizes features for BN learning
4. Learns Bayesian Network per stage
5. Fits parameters (CPTs)
6. Saves the BN locally (JSON)
7. Pushes nodes and edges to Weaviate with confidence scores
"""

import numpy as np
import pandas as pd
import json
import os
from urllib.parse import urlparse
from sklearn.preprocessing import KBinsDiscretizer
from pgmpy.estimators import HillClimbSearch, BIC, MaximumLikelihoodEstimator
import weaviate

# -------------------- CONFIG -------------------- #
STAGES = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
STEP6_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "step6")
BN_SAVE_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "step11")
os.makedirs(BN_SAVE_DIR, exist_ok=True)

DISCRETIZE = True  # Recommended for pgmpy BN
NBINS = 5          # Number of bins for discretization
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_STARTUP_PERIOD = int(os.getenv("WEAVIATE_STARTUP_PERIOD", "30"))
WEAVIATE_TIMEOUT = float(os.getenv("WEAVIATE_TIMEOUT", "5.0"))
WEAVIATE_BEACON_HOST = urlparse(WEAVIATE_URL).hostname or "localhost"

# -------------------- HELPER FUNCTIONS -------------------- #
def aggregate_windows(X_train):
    """
    Aggregate time-series windows to one row per window.
    Options: mean, max, std (we use mean here).
    """
    return X_train.mean(axis=1)  # Shape: (N_windows, N_features)

def discretize_features(df, n_bins=5):
    """
    Discretizes features into ordinal bins for BN learning
    """
    kbd = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    df_discrete = pd.DataFrame(kbd.fit_transform(df), columns=df.columns)
    return df_discrete

def learn_bn(df, score_type='bic'):
    """
    Learn Bayesian Network structure and fit parameters.
    Returns BayesianModel object.
    """
    if score_type.lower() == 'bic':
        scoring = BIC(df)
    else:
        raise ValueError("Only 'bic' score currently supported")
    
    hc = HillClimbSearch(df)
    model = hc.estimate(scoring_method=scoring)
    model.fit(df, estimator=MaximumLikelihoodEstimator)
    return model

def save_bn_to_json(model, feature_names, stage, save_dir=BN_SAVE_DIR):
    """
    Saves BN edges and CPTs to JSON for local storage
    """
    bn_dict = {
        "stage": stage,
        "nodes": list(model.nodes()),
        "edges": [{"from": e[0], "to": e[1]} for e in model.edges()],
        "cpts": {}
    }
    for node in model.nodes():
        cpd = model.get_cpds(node)
        if cpd is None:
            bn_dict["cpts"][node] = None
            continue

        evidence = list(cpd.variables[1:]) if len(cpd.variables) > 1 else []
        evidence_card = [int(v) for v in cpd.cardinality[1:]] if len(cpd.cardinality) > 1 else []

        bn_dict["cpts"][node] = {
            "variable": cpd.variable,
            "variable_card": int(cpd.variable_card),
            "values": cpd.values.reshape(cpd.variable_card, -1).tolist(),
            "evidence": evidence,
            "evidence_card": evidence_card,
        }
    
    save_path = os.path.join(save_dir, f"BN_{stage}.json")
    with open(save_path, 'w') as f:
        json.dump(bn_dict, f, indent=2)
    print(f"[INFO] Saved BN JSON for stage {stage} to {save_path}")
    return save_path

# -------------------- WEAVIATE INTEGRATION -------------------- #
def get_weaviate_client():
    """
    Creates a Weaviate client with retries/startup wait and clear error guidance.
    """
    try:
        client = weaviate.Client(
            WEAVIATE_URL,
            timeout_config=(WEAVIATE_TIMEOUT, WEAVIATE_TIMEOUT),
            startup_period=WEAVIATE_STARTUP_PERIOD,
        )
        if not client.is_ready():
            raise RuntimeError(f"Weaviate at {WEAVIATE_URL} is not ready")
        return client
    except Exception as exc:
        raise RuntimeError(
            f"Could not connect to Weaviate at {WEAVIATE_URL}. "
            "Start Weaviate first, or set WEAVIATE_URL to a running instance."
        ) from exc


def push_bn_to_weaviate(model, feature_names, stage, client):
    """
    Pushes BN nodes and edges to Weaviate
    """
    sensor_nodes = {}
    
    # Create SensorNode objects
    for node in feature_names:
        obj = client.data_object.create(
            data_object={"name": node, "stage": stage},
            class_name="SensorNode"
        )
        sensor_nodes[node] = obj if isinstance(obj, str) else obj["id"]
    
    # Create CausalEdge objects
    for edge in model.edges():
        from_node, to_node = edge
        edge_obj = client.data_object.create(
            data_object={
                "confidence_score": 1.0  # placeholder, can be updated later
            },
            class_name="CausalEdge"
        )
        edge_id = edge_obj if isinstance(edge_obj, str) else edge_obj["id"]

        client.data_object.reference.add(
            from_uuid=edge_id,
            from_class_name="CausalEdge",
            from_property_name="from_sensor",
            to_uuid=sensor_nodes[from_node],
            to_class_name="SensorNode",
        )
        client.data_object.reference.add(
            from_uuid=edge_id,
            from_class_name="CausalEdge",
            from_property_name="to_sensor",
            to_uuid=sensor_nodes[to_node],
            to_class_name="SensorNode",
        )
    print(f"[INFO] Pushed BN for stage {stage} to Weaviate")

# -------------------- MAIN STEP 11 -------------------- #
def step11_learn_bn():
    # Connect to Weaviate
    client = get_weaviate_client()
    
    # Check Weaviate schema
    if not client.schema.exists("SensorNode"):
        print("[INFO] Creating Weaviate schema...")
        schema = {
            "classes": [
                {"class": "SensorNode",
                 "properties": [
                     {"name": "name", "dataType": ["string"]},
                     {"name": "stage", "dataType": ["string"]}
                 ]},
                {"class": "CausalEdge",
                 "properties": [
                     {"name": "from_sensor", "dataType": ["SensorNode"]},
                     {"name": "to_sensor", "dataType": ["SensorNode"]},
                     {"name": "confidence_score", "dataType": ["number"]}
                 ]}
            ]
        }
        client.schema.create(schema)
    
    # Load stage feature map
    with open(os.path.join(STEP6_DIR, "stage_feature_map.json")) as f:
        stage_features_map = json.load(f)
    
    for stage in STAGES:
        print(f"\n[STEP 11] Processing Stage {stage}...")
        
        # Load stage normal training data
        stage_file = os.path.join(STEP6_DIR, f"X_train_{stage}.npy")
        if not os.path.exists(stage_file):
            print(f"[WARNING] Stage {stage} training data not found. Skipping.")
            continue
        
        X_train = np.load(stage_file)
        features = stage_features_map[stage]
        
        if len(X_train) == 0 or len(features) == 0:
            print(f"[WARNING] Stage {stage} has no data/features. Skipping.")
            continue
        
        # Aggregate
        df_stage = pd.DataFrame(aggregate_windows(X_train), columns=features)
        df_stage = df_stage.loc[:, df_stage.std() > 1e-6]  # Remove near-constant features
        # Discretize
        if DISCRETIZE:
            df_stage = discretize_features(df_stage, n_bins=NBINS)
        
        # Learn BN
        bn_model = learn_bn(df_stage)
        print(f"[INFO] Learned BN edges for stage {stage}: {bn_model.edges()}")
        
        # Save locally
        save_bn_to_json(bn_model, features, stage)
        
        # Push to Weaviate
        push_bn_to_weaviate(bn_model, features, stage, client)

if __name__ == "__main__":
    step11_learn_bn()
    print("\n[STEP 11 COMPLETED] Bayesian Networks learned and pushed to Weaviate.")
