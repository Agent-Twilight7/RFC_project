"""
Step 10: RCA Visualization and Inference Reporting.
1. Loads RCA results.
2. Generates Summary Plots (Bar charts of root causes).
3. (Optional) Could generate detailed reconstruction plots for specific incidents.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure output directories exist
os.makedirs("../reports/figures", exist_ok=True)

def plot_rca_summary():
    file_path = "../data/processed/step8/swat_rca_results.csv"
    if not os.path.exists(file_path):
        print("RCA results not found. Run step9.py first.")
        return

    df_rca = pd.read_csv(file_path)
    
    # 1. Guilty Stage Distribution
    plt.figure(figsize=(10, 6))
    df_rca['guilty_stage'].value_counts().plot(kind='bar', color='skyblue')
    plt.title("Distribution of Anomalies by Guilty Stage")
    plt.ylabel("Count of Anomalous Windows")
    plt.xlabel("Stage")
    plt.tight_layout()
    plt.savefig("../reports/figures/step10_rca_stage_dist.png")
    print("Saved stage distribution plot.")
    
    # 2. Top Guilty Features Global
    plt.figure(figsize=(12, 6))
    top_features = df_rca['guilty_feature'].value_counts().head(10)
    top_features.plot(kind='barh', color='salmon')
    plt.title("Top 10 Anomalous Sensors/Actuators (Global)")
    plt.xlabel("Count")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("../reports/figures/step10_rca_feature_top10.png")
    print("Saved feature distribution plot.")
    
    # 3. Top Features Per Stage
    stages = df_rca['guilty_stage'].unique()
    for stage in stages:
        subset = df_rca[df_rca['guilty_stage'] == stage]
        if len(subset) < 10: continue # Skip if trivial
        
        plt.figure(figsize=(10, 5))
        subset['guilty_feature'].value_counts().head(5).plot(kind='bar', color='lightgreen')
        plt.title(f"Top Root Causes for Stage {stage}")
        plt.tight_layout()
        plt.savefig(f"../reports/figures/step10_rca_{stage}_features.png")
        plt.close()

if __name__ == "__main__":
    plot_rca_summary()
    print("Step 10 Completed. Check ../reports/figures/ for visualizations.")
