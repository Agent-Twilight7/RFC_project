"""
Step 8: Inference and Anomaly Detection Plots.
1. Loads trained LSTM models.
2. Runs inference on all data (X_all) to generate reconstruction errors (anomaly scores).
3. Calculates detection thresholds based on Normal data.
4. Plots Anomaly Score Timelines with Thresholds.
5. Plots Actual vs Predicted Reconstructions for top features.
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt

# Project-aware paths (works regardless of current working directory)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
STEP5_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "step5")
STEP6_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "step6")
STEP7_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "step7")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "lstm")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")

# Ensure output directories exist
os.makedirs(STEP7_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Shared Definitions (Must match step7.py) ---
class WindowDataset(Dataset):
    def __init__(self, X):
        self.X = torch.tensor(X, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx]

class SWaT_LSTM_AE(nn.Module):
    def __init__(self, input_dim, seq_len):
        super(SWaT_LSTM_AE, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.enc1 = nn.LSTM(input_size=input_dim, hidden_size=32, batch_first=True)
        self.enc2 = nn.LSTM(input_size=32, hidden_size=16, batch_first=True)
        self.dec1 = nn.LSTM(input_size=16, hidden_size=16, batch_first=True)
        self.dec2 = nn.LSTM(input_size=16, hidden_size=32, batch_first=True)
        self.output_layer = nn.Linear(32, input_dim)
        
    def forward(self, x):
        x_enc1, _ = self.enc1(x)
        _, (h_n, _) = self.enc2(x_enc1)
        bottleneck = h_n[-1]
        x_dec_input = bottleneck.unsqueeze(1).repeat(1, self.seq_len, 1)
        x_dec1, _ = self.dec1(x_dec_input)
        x_dec2, _ = self.dec2(x_dec1)
        x_hat = self.output_layer(x_dec2)
        return x_hat

# --- Plotting Reconstruction ---
def plot_reconstruction_comparison(stage, actual, predicted, feature_names, timestamps, phases):
    # Select first 2 features to avoid clutter
    n_feats = min(actual.shape[1], 2)
    phases = np.array(phases)
    
    for i in range(n_feats):
        feat_name = feature_names[i]
        plt.figure(figsize=(15, 5))
        
        # Plot lines
        plt.plot(timestamps, actual[:, i], label='Actual', color='black', alpha=0.7, linewidth=1)
        plt.plot(timestamps, predicted[:, i], label='Reconstructed', color='cyan', alpha=0.7, linewidth=1) # High contrast
        
        # Shading
        y_min, y_max = plt.ylim()
        
        # Attack Regions (Red)
        attack_mask = np.isin(phases, ['cyber_attack', 'physical_attack'])
        plt.fill_between(timestamps, y_min, y_max, where=attack_mask, color='red', alpha=0.3, label='Attack')
        
        # Normal Regions (Green)
        normal_mask = (phases == 'normal')
        plt.fill_between(timestamps, y_min, y_max, where=normal_mask, color='green', alpha=0.1, label='Normal')
        
        plt.title(f"Reconstruction: Stage {stage} - {feat_name}")
        plt.legend(loc='upper right')
        plt.tight_layout()
        save_path = os.path.join(FIGURES_DIR, f"step8_recon_{stage}_{feat_name}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved reconstruction plot to {save_path}")

# --- Inference Function ---
def run_inference_and_score(stage, timestamps, phases):
    print(f"\nProcessing Inference for Stage {stage}...")
    
    # Paths
    model_path = os.path.join(MODEL_DIR, f"AE_{stage}_best.pt")
    data_path = os.path.join(STEP6_DIR, f"X_all_{stage}.npy")
    
    if not os.path.exists(model_path) or not os.path.exists(data_path):
        print(f"Missing model or data for {stage}. Skipping.")
        return None
        
    # Load Data
    X_all = np.load(data_path)
    seq_len = X_all.shape[1]
    input_dim = X_all.shape[2]
    
    # Load Model
    model = SWaT_LSTM_AE(input_dim, seq_len).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
        
    model.eval()
    
    # Run Prediction
    dataset = WindowDataset(X_all)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    reconstructions = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            recon = model(data)
            reconstructions.append(recon.cpu().numpy())
    
    reconstructions = np.concatenate(reconstructions, axis=0)
    
    # --- Feature Names ---
    cols = [f"Feat{i}" for i in range(input_dim)]
    meta_path = os.path.join(STEP6_DIR, "stage_feature_map.json")
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            loaded_cols = meta.get(stage, [])
            if len(loaded_cols) == input_dim:
                cols = loaded_cols

    # --- Plot Predictions vs Actual (Last Point) ---
    print("Plotting actual vs predicted...")
    actual_last = X_all[:, -1, :]
    recon_last = reconstructions[:, -1, :]
    plot_reconstruction_comparison(stage, actual_last, recon_last, cols, timestamps, phases)
    
    # --- Calculate Score ---
    # Calculate MSE (Mean Over Time) per Feature
    mse = np.mean((X_all - reconstructions) ** 2, axis=1)
    
    # Save to DataFrame
    df_scores = pd.DataFrame(mse, columns=cols)
    
    save_path = os.path.join(STEP7_DIR, f"swat_{stage}_anomaly_scores.csv")
    df_scores.to_csv(save_path, index=False)
    print(f"Saved scores to {save_path}")
    return df_scores

# --- Plotting Function ---
def plot_timeline(df_res, stage, threshold):
    plt.figure(figsize=(15, 6))
    score_col = f'score_{stage}'
    
    if score_col not in df_res.columns: return

    plt.plot(df_res.index, df_res[score_col], label=f'MSE ({stage})', color='black', alpha=0.7, linewidth=1)
    
    # Threshold
    if threshold > 0:
        plt.axhline(threshold, color='blue', linestyle='--', label=f'Threshold ({threshold:.4f})')
    
    plt.title(f"Anomaly Detection Timeline - Stage {stage}")
    plt.ylabel("Reconstruction Error (MSE)")
    plt.legend(loc='upper right')
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, f"step8_timeline_{stage}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")

# --- Main ---
if __name__ == "__main__":
    stages = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
    
    # 1. Load Metadata for Time Indexing
    print("Loading time metadata...")
    window_end_times = np.load(os.path.join(STEP5_DIR, "window_end_times.npy"), allow_pickle=True)
    window_phase = np.load(os.path.join(STEP5_DIR, "window_phase.npy"), allow_pickle=True)
    
    df_res = pd.DataFrame({'t_stamp': window_end_times, 'phase': window_phase})
    df_res['t_stamp'] = pd.to_datetime(df_res['t_stamp'])
    df_res = df_res.set_index('t_stamp')

    # 2. Run Inference & Collect Scores
    for stage in stages:
        # Pass timestamps index and phases for plotting
        df_feat_scores = run_inference_and_score(stage, df_res.index, df_res['phase'])
        
        if df_feat_scores is not None:
            # Aggregate to Stage Score (Mean across features)
            stage_score = df_feat_scores.mean(axis=1).values
            df_res[f'score_{stage}'] = stage_score

    # 3. Calculate Thresholds & Plot
    print("\nCalculating Thresholds and Plotting...")
    normal_mask = (df_res['phase'] == 'normal')
    
    for stage in stages:
        col = f'score_{stage}'
        if col not in df_res.columns: continue
        
        # Threshold (99.9th percentile of Normal)
        normal_scores = df_res.loc[normal_mask, col]
        if len(normal_scores) > 0:
            threshold = normal_scores.quantile(0.999)
        else:
            threshold = 0.0
            
        print(f"{stage} Threshold: {threshold:.6f}")
        
        # Plot
        plot_timeline(df_res, stage, threshold)
        
    print("\nStep 8 Completed. Scores generated and timelines plotted.")

