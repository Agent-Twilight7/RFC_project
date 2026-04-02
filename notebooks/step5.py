"""
Loads the normalized data and creates sliding windows (sequences) of data for 
time-series analysis, separating the data into training (normal) and all windows.
"""
import pandas as pd
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
STEP4_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "step4")
STEP5_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "step5")

# Ensure output directories exist
os.makedirs(STEP5_DIR, exist_ok=True)

# 1. Load Normalized Data
input_path = os.path.join(STEP4_DIR, "swat_step4_normalized.csv")
if not os.path.exists(input_path):
    raise FileNotFoundError(
        f"Missing required input: {input_path}. Run step4.py first to generate normalized data."
    )
print(f"Loading {input_path}...")
df = pd.read_csv(input_path)

if 't_stamp' in df.columns:
    df['t_stamp'] = pd.to_datetime(df['t_stamp'])
    df = df.set_index('t_stamp')

# 2. Sliding Window Parameters
WINDOW_SIZE = 60      # seconds
STRIDE = 1            # seconds

def create_sliding_windows(df, window_size, stride):
    X = []
    window_start_times = []
    window_end_times = []

    values = df.values
    index = df.index

    # Note: This loop can be slow in Python.
    # For very large datasets, use stride_tricks or pre-indexing.
    # Given dataset size (~20-50k rows), this is acceptable.
    for start in range(0, len(df) - window_size + 1, stride):
        end = start + window_size

        # Check for continuous timestamps (1-second sampling)
        expected_duration = pd.Timedelta(seconds=window_size - 1)
        
        # Optimization: Pre-check if indices are valid
        if index[end - 1] - index[start] != expected_duration:
            continue

        X.append(values[start:end])
        window_start_times.append(index[start])
        window_end_times.append(index[end - 1])

    if len(X) == 0:
        return np.array([]), np.array([]), np.array([])
        
    X = np.stack(X)
    return X, np.array(window_start_times), np.array(window_end_times)

print("Creating sliding windows...")
X_all, win_start, win_end = create_sliding_windows(
    df,
    WINDOW_SIZE,
    STRIDE
)
print("Windowed tensor shape:", X_all.shape)

# 3. Label Windows (Phases)
NORMAL_END = pd.Timestamp("2019-12-06 10:20:00")
CYBER_ATTACK_START = pd.Timestamp("2019-12-06 10:30:00")
CYBER_ATTACK_END   = pd.Timestamp("2019-12-06 11:20:00")
PHYSICAL_ATTACK_START = pd.Timestamp("2019-12-06 12:30:00")
PHYSICAL_ATTACK_END   = pd.Timestamp("2019-12-06 13:25:00")

window_phase = np.full(len(win_start), "neutral", dtype=object)

# Normal: Ends before normal period ends
window_phase[win_end < NORMAL_END] = "normal"

# Cyber Attack: Fully contained within cyber attack period
# (Can adjust logic to 'overlaps' if preferred, but 'contained' is safer for pure labels)
window_phase[
    (win_start <= CYBER_ATTACK_END) &
    (win_end >= CYBER_ATTACK_START)
] = "cyber_attack"

# Physical Attack
window_phase[
    (win_start <= PHYSICAL_ATTACK_END) &
    (win_end >= PHYSICAL_ATTACK_START)
] = "physical_attack"

print("Window Phase Counts:")
print(pd.Series(window_phase).value_counts())

# 4. Filter Training Set (Only Normal)
X_train = X_all[window_phase == "normal"]
print("Training windows shape:", X_train.shape)

# 5. Save Arrays
print("Saving arrays...")
np.save(os.path.join(STEP5_DIR, "X_all_windows.npy"), X_all)
np.save(os.path.join(STEP5_DIR, "window_phase.npy"), window_phase)
np.save(os.path.join(STEP5_DIR, "window_start_times.npy"), win_start)
np.save(os.path.join(STEP5_DIR, "window_end_times.npy"), win_end)
np.save(os.path.join(STEP5_DIR, "X_train_windows.npy"), X_train)
print(f"Saved all arrays to {STEP5_DIR}")
