"""
Loads the cleaned step 3 data, calculates mean and standard deviation from the normal 
training period, and normalizes the Pv (Process Variable) columns using Z-score normalization.
"""
import pandas as pd
import numpy as np
import os

# Ensure output directories exist
os.makedirs("../data/processed/step4", exist_ok=True)

# 1. Load Step 3 Data
input_path = "../data/processed/step3/swat_step3_clean.csv"
print(f"Loading {input_path}...")
df_step3 = pd.read_csv(input_path)

# Convert t_stamp back to datetime and index
if 't_stamp' in df_step3.columns:
    df_step3['t_stamp'] = pd.to_datetime(df_step3['t_stamp'])
    df_step3 = df_step3.set_index('t_stamp')

# Load Metadata to identify Pv columns
meta_path = "../data/processed/step2/swat_feature_metadata.csv"
meta_df = pd.read_csv(meta_path)

pv_cols = meta_df.loc[meta_df['type'] == 'Pv', 'column']
pv_cols = [c for c in pv_cols if c in df_step3.columns]
print(f"Found {len(pv_cols)} Pv columns to normalize.")

# 2. Define Training Split for Normalization
NORMAL_END = pd.Timestamp("2019-12-06 10:20:00")
df_step3 = df_step3.sort_index()

df_train_norm = df_step3.loc[df_step3.index < NORMAL_END]
print("Training data for scaler shape:", df_train_norm.shape)

# 3. Compute Mean and Std
pv_mean = df_train_norm[pv_cols].mean()
pv_std  = df_train_norm[pv_cols].std()

# Replace 0.0 standard deviations with 1.0 to avoid division by zero
# (Constant columns become 0 after normalization)
pv_std = pv_std.replace(0.0, 1.0)

# 4. Normalize
df_step4 = df_step3.copy()
print("First 5 rows of Pv before normalization:")
print(df_step4[pv_cols].iloc[0:5].head())

df_step4[pv_cols] = (df_step4[pv_cols] - pv_mean) / pv_std

print("First 5 rows of Pv after normalization:")
print(df_step4[pv_cols].iloc[0:5].head())

# 5. Save Scaler and Normalized Data
scaler = {
    'mean': pv_mean,
    'std': pv_std
}
scaler_path = "../data/processed/step4/swat_pv_scaler.pkl"
pd.to_pickle(scaler, scaler_path)
print(f"Saved scaler to {scaler_path}")

output_path = "../data/processed/step4/swat_step4_normalized.csv"
df_step4.to_csv(output_path)
print(f"Saved normalized data to {output_path}")
