"""
Loads the sliding windows and metadata, maps features to their respective stages (P1-P6), 
and splits the windowed datasets by stage.
"""
import numpy as np
import pandas as pd
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
STEP2_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "step2")
STEP4_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "step4")
STEP5_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "step5")
STEP6_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "step6")

# Ensure output directories exist
os.makedirs(STEP6_DIR, exist_ok=True)

# 1. Load Data
print("Loading sliding windows and metadata...")
step5_x_all_path = os.path.join(STEP5_DIR, "X_all_windows.npy")
step5_x_train_path = os.path.join(STEP5_DIR, "X_train_windows.npy")
meta_path = os.path.join(STEP2_DIR, "swat_feature_metadata.csv")
step4_path = os.path.join(STEP4_DIR, "swat_step4_normalized.csv")

required_inputs = [
    (step5_x_all_path, "Run step5.py first"),
    (step5_x_train_path, "Run step5.py first"),
    (meta_path, "Run step2.py first"),
    (step4_path, "Run step4.py first"),
]
for file_path, hint in required_inputs:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing required input: {file_path}. {hint}.")

X_all = np.load(step5_x_all_path)
X_train = np.load(step5_x_train_path)

meta_df = pd.read_csv(meta_path)
df_step4 = pd.read_csv(step4_path)

# Identify feature columns (excluding t_stamp)
feature_columns = [c for c in df_step4.columns if c != 't_stamp']
col_to_idx = {col: i for i, col in enumerate(feature_columns)}

# Sanity check
assert X_all.shape[2] == len(feature_columns), "Feature dimension mismatch!"

# 2. Map Features to Stages
stage_features = {}
stages = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']

for stage in stages:
    # Select columns for this stage that are Pv (ONLY Pv)
    cols = meta_df.loc[
        (meta_df['stage'] == stage) &
        (meta_df['type'] == 'Pv'),
        'column'
    ].tolist()

    # Keep only those present in the final feature list
    cols = [c for c in cols if c in col_to_idx]
    stage_features[stage] = cols

# Create index mapping
stage_feature_indices = {
    stage: [col_to_idx[c] for c in cols]
    for stage, cols in stage_features.items()
}

# 3. Split Arrays by Stage
X_all_stage = {}
X_train_stage = {}

print("Splitting datasets by stage...")
for stage, idxs in stage_feature_indices.items():
    X_all_stage[stage] = X_all[:, :, idxs]
    X_train_stage[stage] = X_train[:, :, idxs]
    
    print(f"{stage}: All={X_all_stage[stage].shape}, Train={X_train_stage[stage].shape}")

# 4. Save
for stage in X_all_stage:
    np.save(os.path.join(STEP6_DIR, f"X_all_{stage}.npy"), X_all_stage[stage])
    np.save(os.path.join(STEP6_DIR, f"X_train_{stage}.npy"), X_train_stage[stage])

# Save mappings
with open(os.path.join(STEP6_DIR, "stage_feature_map.json"), "w") as f:
    json.dump(stage_features, f, indent=2)

print(f"Saved per-stage arrays and feature map to {STEP6_DIR}")
