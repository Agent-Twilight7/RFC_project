"""
Loads the raw SWaT dataset, processes timestamps, analyzes time gaps, defines attack periods 
(Normal, Cyber Attack, Physical Attack), and creates masks for these stages.
"""
import pandas as pd
import matplotlib.pyplot as plt
import os

# Ensure output directories exist
# os.makedirs("../data/processed", exist_ok=True)

# 1. Load historian data
# Adjust path as necessary
data_path = r"../data/processed/swat_cleaned.csv"
if not os.path.exists(data_path):
    # Fallback for running from different cwd
    data_path = r"data/processed/swat_cleaned.csv"

print(f"Loading data from {data_path}...")
df = pd.read_csv(data_path)
print("Columns:", df.columns)

# 2. Process timestamps
df['t_stamp'] = pd.to_datetime(df['t_stamp'])
df = df.sort_values('t_stamp').reset_index(drop=True)
df = df.set_index('t_stamp')

print(f"Date range: {df.index.min()} to {df.index.max()}")

# 3. Analyze time gaps
time_diffs = df.index.to_series().diff().dropna()
print("Time differences head:")
print(time_diffs.value_counts().head())

# 4. Define Attack Periods
NORMAL_END = pd.Timestamp("2019-12-06 10:20:00")

CYBER_ATTACK_START = pd.Timestamp("2019-12-06 10:30:00")
CYBER_ATTACK_END   = pd.Timestamp("2019-12-06 11:20:00")

PHYSICAL_ATTACK_START = pd.Timestamp("2019-12-06 12:30:00")
PHYSICAL_ATTACK_END   = pd.Timestamp("2019-12-06 13:25:00")

POST_ATTACK_START = pd.Timestamp("2019-12-06 13:30:00")

# 5. Create Stage Masks
mask_normal = df.index < NORMAL_END
mask_cyber_attack = (df.index >= CYBER_ATTACK_START) & (df.index <= CYBER_ATTACK_END)
mask_physical_attack = (df.index >= PHYSICAL_ATTACK_START) & (df.index <= PHYSICAL_ATTACK_END)
mask_post_attack = df.index >= POST_ATTACK_START

print("Normal points:", mask_normal.sum())
print("Cyber attack points:", mask_cyber_attack.sum())
print("Physical attack points:", mask_physical_attack.sum())
print("Post attack points:", mask_post_attack.sum())

# 6. Visualization
# Select a few 'Pv' (Process value) and 'Status' columns to plot sample data
plot_cols = [c for c in df.columns if 'Pv' in c][:2] + [c for c in df.columns if 'Status' in c][:1]

if not plot_cols:
    plot_cols = df.columns[:3]

plt.figure(figsize=(15, 3 * len(plot_cols)))

for i, col in enumerate(plot_cols):
    plt.subplot(len(plot_cols), 1, i+1)
    plt.plot(df.index, df[col], label='Value', linewidth=1, color='tab:blue')
    
    # Shade the different phases
    plt.axvspan(df.index.min(), NORMAL_END, color='green', alpha=0.1, label='Normal' if i==0 else None)
    plt.axvspan(CYBER_ATTACK_START, CYBER_ATTACK_END, color='red', alpha=0.1, label='Cyber Attack' if i==0 else None)
    plt.axvspan(PHYSICAL_ATTACK_START, PHYSICAL_ATTACK_END, color='orange', alpha=0.1, label='Physical Attack' if i==0 else None)
    
    plt.title(col)
    plt.grid(True, alpha=0.3)
    if i == 0:
        plt.legend(loc='upper right')

plt.tight_layout()
print("Showing plot...")
plt.show() # Commented out for non-interactive execution
