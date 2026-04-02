"""
Step 9: Root Cause Analysis (RCA) Computation.
1. Loads anomaly scores and thresholds (re-calculated).
2. Identifies 'Guilty Stage' for each anomalous window (Stage-level RCA).
3. Identifies 'Guilty Feature' within the guilty stage (Feature-level RCA).
4. Saves the detailed RCA logs to CSV.
"""
import pandas as pd
import numpy as np
import os
import json

# Project-aware paths (works regardless of current working directory)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
STEP5_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "step5")
STEP7_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "step7")
STEP8_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "step8")

# Ensure output directories exist
os.makedirs(STEP8_DIR, exist_ok=True)


def save_step12_anomalies_json(rca_results, output_path):
    anomalies = []
    for idx, result in enumerate(rca_results):
        anomalous_sensor_scores = result.get("anomalous_sensor_scores", {})
        anomalous_sensors = result.get("anomalous_sensors", [])
        guilty_feature = str(result.get("guilty_feature", ""))
        if not anomalous_sensors and guilty_feature:
            anomalous_sensors = [guilty_feature]

        if not anomalous_sensor_scores and guilty_feature:
            guilty_score = result.get("feature_score", 0.0)
            anomalous_sensor_scores = {guilty_feature: float(guilty_score)}

        anomalies.append(
            {
                "id": f"anom_{idx}",
                "stage": str(result["guilty_stage"]),
                "deviating_sensors": anomalous_sensors,
                "evidence_bins": {},
                "anomalous_sensors": anomalous_sensors,
                "anomalous_sensor_scores": anomalous_sensor_scores,
                "guilty_feature": guilty_feature,
                "guilty_feature_score": float(result.get("feature_score", 0.0)),
                "timestamp": str(result["t_stamp"]),
                "phase": str(result["phase"]),
            }
        )

    with open(output_path, "w") as f:
        json.dump(anomalies, f, indent=2)

def perform_rca():
    print("Loading Data for RCA...")
    
    # 1. Load Time & Phase
    window_end_times = np.load(os.path.join(STEP5_DIR, "window_end_times.npy"), allow_pickle=True)
    window_phase = np.load(os.path.join(STEP5_DIR, "window_phase.npy"), allow_pickle=True)
    
    df_res = pd.DataFrame({'t_stamp': window_end_times, 'phase': window_phase})
    df_res['t_stamp'] = pd.to_datetime(df_res['t_stamp'])
    df_res = df_res.set_index('t_stamp')
    
    stages = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
    stage_feat_scores = {} # Dict to hold DF of feature scores per stage
    thresholds = {}
    feature_thresholds = {}
    
    # 2. Load Scores & Calc Thresholds
    normal_mask = (df_res['phase'] == 'normal')
    
    for stage in stages:
        path = os.path.join(STEP7_DIR, f"swat_{stage}_anomaly_scores.csv")
        if not os.path.exists(path):
            continue
            
        df_s = pd.read_csv(path)
        stage_feat_scores[stage] = df_s
        
        # Stage Score = Mean of features
        stage_mean = df_s.mean(axis=1).values
        df_res[f'score_{stage}'] = stage_mean
        
        # Threshold
        normal_vals = stage_mean[normal_mask]
        thresh = np.quantile(normal_vals, 0.999) if len(normal_vals) > 0 else 0
        thresholds[stage] = thresh

        normal_indices = np.where(normal_mask.to_numpy())[0]
        if len(normal_indices) > 0:
            normal_df = df_s.iloc[normal_indices]
        else:
            normal_df = df_s
        feature_thresholds[stage] = normal_df.quantile(0.999)
        
        # Mark Anomaly
        df_res[f'is_anomaly_{stage}'] = df_res[f'score_{stage}'] > thresh

    # Global Anomaly
    anomaly_cols = [c for c in df_res.columns if 'is_anomaly_' in c]
    df_res['is_anomaly_global'] = df_res[anomaly_cols].any(axis=1) if anomaly_cols else False

    if not stage_feat_scores:
        print("No stage anomaly score files found. Run step8.py first.")
        output_csv = os.path.join(STEP8_DIR, "swat_rca_results.csv")
        output_json = os.path.join(STEP8_DIR, "anomalies.json")
        pd.DataFrame(
            columns=[
                "t_stamp",
                "phase",
                "guilty_stage",
                "max_stage_ratio",
                "guilty_feature",
                "feature_score",
                "anomalous_sensors",
                "anomalous_sensor_scores",
            ]
        ).to_csv(output_csv, index=False)
        with open(output_json, "w") as f:
            json.dump([], f, indent=2)
        print(f"Saved empty RCA CSV to {output_csv}")
        print(f"Saved empty anomalies JSON to {output_json}")
        return
    
    print(f"Total Anomalies Detected: {df_res['is_anomaly_global'].sum()}")
    
    # 3. RCA Logic
    print("Running RCA Loop...")
    rca_results = []
    
    # Pre-computation: Index mapping
    # df_res index corresponds to integer index 0..N
    # We need integer index to access stage_feat_scores dict DFs
    
    # Iterate using integer index for speed and direct access
    # Get boolean array
    is_anom = df_res['is_anomaly_global'].values
    indices = np.where(is_anom)[0]
    
    for i in indices:
        row = df_res.iloc[i]
        t_stamp = df_res.index[i]
        
        # Level 1: Find Guilty Stage (Max Ratio)
        ratios = {}
        for stage in stages:
            if f'score_{stage}' not in row: continue
            
            # Check if this stage contributed to the anomaly (exceeds threshold)
            # If we want purely "max ratio", we can check all, but it makes sense 
            # to only blame stages that are actually anomalous.
            if row[f'is_anomaly_{stage}']:
                threshold = thresholds.get(stage, 0)
                if threshold > 0:
                    ratios[stage] = row[f'score_{stage}'] / threshold
        
        if not ratios:
            # Determine fallback: maybe one stage was VERY close? 
            # Or just take max ratio even if not crossing threshold (edge case)
            continue
            
        guilty_stage = max(ratios, key=ratios.get)
        max_ratio = ratios[guilty_stage]
        
        # Level 2: Find Guilty Feature in that Stage
        # Access the feature scores DF
        feat_scores_row = stage_feat_scores[guilty_stage].iloc[i]
        guilty_feature = feat_scores_row.idxmax()
        feature_val = float(feat_scores_row.max())

        stage_thresholds = feature_thresholds.get(guilty_stage)
        anomalous_sensor_scores = {}
        if stage_thresholds is not None:
            for sensor_name, sensor_score in feat_scores_row.items():
                sensor_threshold = float(stage_thresholds.get(sensor_name, np.inf))
                sensor_score_value = float(sensor_score)
                if sensor_score_value > sensor_threshold:
                    anomalous_sensor_scores[str(sensor_name)] = sensor_score_value

        if not anomalous_sensor_scores:
            top_scores = feat_scores_row.sort_values(ascending=False).head(3)
            anomalous_sensor_scores = {
                str(sensor_name): float(sensor_score)
                for sensor_name, sensor_score in top_scores.items()
            }

        anomalous_sensors = list(anomalous_sensor_scores.keys())
        
        rca_results.append({
            't_stamp': t_stamp,
            'phase': row['phase'],
            'guilty_stage': guilty_stage,
            'max_stage_ratio': max_ratio,
            'guilty_feature': guilty_feature,
            'feature_score': feature_val,
            'anomalous_sensors': anomalous_sensors,
            'anomalous_sensor_scores': anomalous_sensor_scores,
        })
        
    # 4. Save
    df_rca = pd.DataFrame(rca_results)
    output_csv = os.path.join(STEP8_DIR, "swat_rca_results.csv")
    output_json = os.path.join(STEP8_DIR, "anomalies.json")

    if not df_rca.empty:
        df_rca['anomalous_sensors'] = df_rca['anomalous_sensors'].apply(json.dumps)
        df_rca['anomalous_sensor_scores'] = df_rca['anomalous_sensor_scores'].apply(json.dumps)

    df_rca.to_csv(output_csv, index=False)
    save_step12_anomalies_json(rca_results, output_json)

    print(f"RCA Results saved to {output_csv}")
    print(f"Anomalies JSON saved to {output_json}")

    if df_rca.empty:
        print("No anomalous windows passed RCA filters.")
    else:
        print(f"Top Causes:\n{df_rca.groupby(['guilty_stage', 'guilty_feature']).size().nlargest(5)}")

if __name__ == "__main__":
    perform_rca()
