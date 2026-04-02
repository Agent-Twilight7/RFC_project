"""
Loads the cleaned data and defines helper functions to classify column types 
(Pv, Status, Alarm, State) and extract stages (P1-P6) from column names.
"""
import pandas as pd
import re
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Ensure output directories exist
os.makedirs(os.path.join(project_root, "data/processed/step2"), exist_ok=True)

# 1. Load Data
data_path = os.path.join(project_root, "data/processed/swat_cleaned.csv")
print(f"Loading data from {data_path}...")
df_clean = pd.read_csv(data_path)
cols = df_clean.columns.tolist()
print(f"Total columns: {len(cols)}")

# 2. Define Helper Functions
def classify_type(col):
    if col.endswith('.Pv'):
        return 'Pv'
    elif col.endswith('.Status'):
        return 'Status'
    elif col.endswith('.Alarm'):
        return 'Alarm'
    elif re.match(r'^P[1-6]_STATE$', col):
        return 'State'
    else:
        return 'Other'

def extract_stage(col):
    # P*_STATE
    m = re.match(r'^P([1-6])_STATE$', col)
    if m:
        return f'P{m.group(1)}'
    
    # Tags like LIT101, MV301, FIT601, etc.
    m = re.search(r'(\d{3})', col)
    if m:
        block = int(m.group(1))
        if 100 <= block < 200:
            return 'P1'
        elif 200 <= block < 300:
            return 'P2'
        elif 300 <= block < 400:
            return 'P3'
        elif 400 <= block < 500:
            return 'P4'
        elif 500 <= block < 600:
            return 'P5'
        elif 600 <= block < 700:
            return 'P6'
    return 'Unknown'

# 3. Generate Metadata
meta = []

for col in cols:
    meta.append({
        'column': col,
        'type': classify_type(col),
        'stage': extract_stage(col)
    })

meta_df = pd.DataFrame(meta)
print(meta_df.head(10))

print("\nValue Counts (Type):")
print(meta_df['type'].value_counts())

print("\nValue Counts (Stage):")
print(meta_df['stage'].value_counts())

# 4. Save Metadata
output_path = "../data/processed/step2/swat_feature_metadata.csv"
meta_df.to_csv(output_path, index=False)
print(f"Saved feature metadata to {output_path}")
