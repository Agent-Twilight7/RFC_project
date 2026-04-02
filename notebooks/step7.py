"""
Trains LSTM Autoencoders for each stage (P1-P6) using strictly normal data.
Architecture: LSTM(32) -> LSTM(16) -> Bottleneck -> LSTM(16) -> LSTM(32) -> Dense.
Saves models and calculates reconstruction errors for anomaly detection.
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import json
import os
import copy
import sys

# Ensure output directories exist
os.makedirs("../models/lstm", exist_ok=True)
os.makedirs("../data/processed/step7", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Dataset Definition ---
class WindowDataset(Dataset):
    def __init__(self, X):
        self.X = torch.tensor(X, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx]

# --- Model Definition ---
class SWaT_LSTM_AE(nn.Module):
    def __init__(self, input_dim, seq_len):
        super(SWaT_LSTM_AE, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        
        # Encoder
        self.enc1 = nn.LSTM(input_size=input_dim, hidden_size=32, batch_first=True)
        self.enc2 = nn.LSTM(input_size=32, hidden_size=16, batch_first=True)
        
        # Decoder
        self.dec1 = nn.LSTM(input_size=16, hidden_size=16, batch_first=True)
        self.dec2 = nn.LSTM(input_size=16, hidden_size=32, batch_first=True)
        
        # Output
        self.output_layer = nn.Linear(32, input_dim)
        
    def forward(self, x):
        # Encoder
        x_enc1, _ = self.enc1(x)
        _, (h_n, _) = self.enc2(x_enc1)
        bottleneck = h_n[-1] # (B, 16)
        
        # Decoder Input
        x_dec_input = bottleneck.unsqueeze(1).repeat(1, self.seq_len, 1)
        
        # Decoder
        x_dec1, _ = self.dec1(x_dec_input)
        x_dec2, _ = self.dec2(x_dec1)
        x_hat = self.output_layer(x_dec2)
        return x_hat

# --- Training Loop ---
def train_stage_model(stage, X_train):
    print(f"\nTraining Model for Stage: {stage}")
    if len(X_train) == 0:
        print("No training data!")
        return None
        
    seq_len = X_train.shape[1]
    input_dim = X_train.shape[2]
    
    # Validation Split
    val_size = int(len(X_train) * 0.1)
    train_size = len(X_train) - val_size
    
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    train_idx, val_idx = indices[:train_size], indices[train_size:]
    
    X_t = X_train[train_idx]
    X_v = X_train[val_idx]
    
    train_loader = DataLoader(WindowDataset(X_t), batch_size=32, shuffle=True)
    val_loader = DataLoader(WindowDataset(X_v), batch_size=32, shuffle=False)
    
    model = SWaT_LSTM_AE(input_dim, seq_len).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 40 
    patience = 5
    best_val_loss = float('inf')
    counter = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                output = model(data)
                loss = criterion(output, data)
                val_loss += loss.item() * data.size(0)
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                # print("Early stopping triggered.")
                break
                
    model.load_state_dict(best_model_wts)
    save_path = f"../models/lstm/AE_{stage}_best.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Saved model to {save_path}")
    return model

# --- Inference ---
def run_inference(model, X_all, stage):
    if model is None: return
    
    model.eval()
    dataset = WindowDataset(X_all)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    reconstructions = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            recon = model(data)
            reconstructions.append(recon.cpu().numpy())
            
    reconstructions = np.concatenate(reconstructions, axis=0)
    mse = np.mean((X_all - reconstructions) ** 2, axis=1) # Mean over time -> (N_windows, N_features)
    
    df_scores = pd.DataFrame(mse)
    meta_path = "../data/processed/step6/stage_feature_map.json"
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            cols = meta.get(stage, [])
            if len(cols) == mse.shape[1]:
                df_scores.columns = cols
    
    save_path = f"../data/processed/step7/swat_{stage}_anomaly_scores.csv"
    df_scores.to_csv(save_path, index=False)
    print(f"Saved anomaly scores to {save_path}")

if __name__ == "__main__":
    stages = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
    for stage in stages:
        train_file = f"../data/processed/step6/X_train_{stage}.npy"
        all_file = f"../data/processed/step6/X_all_{stage}.npy"
        
        if not os.path.exists(train_file):
            print(f"Data for {stage} not found. Skipping.")
            continue
            
        print(f"Processing Stage {stage}...")
        X_train = np.load(train_file)
        X_all = np.load(all_file)
        
        model = train_stage_model(stage, X_train)
        run_inference(model, X_all, stage)
