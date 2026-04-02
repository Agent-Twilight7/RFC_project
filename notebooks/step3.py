"""
Loads data and metadata, drops alarm columns, converts column types (Pv to float, 
Status/State to proper integer types), and imputes missing values using forward and backward fill.
"""
import pandas as pd
import os

# Ensure output directories exist
os.makedirs("../data/processed/step3", exist_ok=True)

# 1. Load Data and Metadata
data_path = r"../data/processed/swat_cleaned.csv"
meta_path = r"../data/processed/step2/swat_feature_metadata.csv"

print("Loading data...")
df_clean = pd.read_csv(data_path)
# Ensure timestamp is datetime
df_clean['t_stamp'] = pd.to_datetime(df_clean['t_stamp'])
df_clean = df_clean.sort_values('t_stamp').reset_index(drop=True)
df_clean = df_clean.set_index('t_stamp')

print("Loading metadata...")
meta_df = pd.read_csv(meta_path)

# 2. Drop Alarm Columns
alarm_cols = meta_df.loc[meta_df['type'] == 'Alarm', 'column'].tolist()
df_step3 = df_clean.drop(columns=alarm_cols)
print(f"Dropped {len(alarm_cols)} Alarm columns.")

# 3. Convert Types
# Pv -> Float
pv_cols = meta_df.loc[meta_df['type'] == 'Pv', 'column']
pv_cols = [c for c in pv_cols if c in df_step3.columns]
df_step3[pv_cols] = df_step3[pv_cols].astype(float)
print(f"Converted {len(pv_cols)} Pv columns to float.")

# Status -> Int64 (nullable int)
status_cols = meta_df.loc[meta_df['type'] == 'Status', 'column']
status_cols = [c for c in status_cols if c in df_step3.columns]
df_step3[status_cols] = df_step3[status_cols].astype('Int64')
print(f"Converted {len(status_cols)} Status columns to Int64.")

# State -> int
state_cols = meta_df.loc[meta_df['type'] == 'State', 'column']
state_cols = [c for c in state_cols if c in df_step3.columns]
df_step3[state_cols] = df_step3[state_cols].astype(int)
print(f"Converted {len(state_cols)} State columns to int.")

# 4. Check & Impute Missing Values
missing_before = df_step3.isna().sum().sum()
print(f"Missing values before imputation: {missing_before}")

# Impute (Forward Fill then Backward Fill)
df_step3 = df_step3.ffill().bfill()

missing_after = df_step3.isna().sum().sum()
print(f"Missing values after imputation: {missing_after}")

# 5. Save Cleaned Data
output_path = "../data/processed/step3/swat_step3_clean.csv"
# The index (t_stamp) will be saved
df_step3.to_csv(output_path)
print(f"Saved cleaned data to {output_path}")
