# RESULTS

## Purpose Of This Document

This file documents the actual outputs already generated in this repository.
It focuses on:

- what result files exist
- what the main plots show
- what the current anomaly and RCA outputs look like
- where to find each artifact in the directory tree
- how to inspect the visual outputs systematically

For data meaning, see `INFO.md`.
For step-by-step code behavior, see `CODE.md`.

## Result Artifact Overview

The repository currently contains generated result artifacts in a few main places:

1. `data/processed/step7` through `data/processed/step15`
2. `reports/figures`
3. `reports/figures/bn_graphs`

These outputs correspond to several analysis layers:

- reconstruction-based anomaly detection
- reconstruction-based root cause analysis
- Bayesian Network learning and Bayesian root cause analysis
- local LLM-based evaluation of propagation path plausibility
- temporal consistency evaluation and visualization for propagation paths

## High-Level Numerical Summary

The current generated RCA outputs show:

- reconstruction-based RCA rows: `12306`
- Bayesian RCA rows in step 12: `12306`

That means the step 12 Bayesian layer processed the same anomaly candidate set produced earlier by the reconstruction-error RCA stage.

## Stage Distribution Of Detected Anomalies

From `data/processed/step8/swat_rca_results.csv`, the current guilty-stage counts are:

| Stage | Count |
|---|---:|
| `P6` | 5181 |
| `P3` | 4171 |
| `P2` | 2229 |
| `P5` | 423 |
| `P1` | 301 |
| `P4` | 1 |

## Top Global Guilty Features

From the same RCA output, the most frequent guilty features are:

| Rank | Feature | Count |
|---|---|---:|
| 1 | `FIT601.Pv` | 5181 |
| 2 | `AIT303.Pv` | 2335 |
| 3 | `FIT201.Pv` | 2066 |
| 4 | `FIT301.Pv` | 1830 |
| 5 | `AIT503.Pv` | 412 |
| 6 | `FIT101.Pv` | 167 |
| 7 | `AIT201.Pv` | 163 |
| 8 | `LIT101.Pv` | 134 |
| 9 | `PIT502.Pv` | 11 |
| 10 | `DPIT301.Pv` | 6 |

## Phase Distribution Of RCA Rows

The current RCA rows are distributed across phases as follows:

| Phase | Count |
|---|---:|
| `neutral` | 5881 |
| `physical_attack` | 3360 |
| `cyber_attack` | 3060 |
| `normal` | 5 |

## Bayesian Network Structure Summary

The current learned Bayesian Networks under `data/processed/step11` have these sizes:

| Stage | Nodes | Edges |
|---|---:|---:|
| `P1` | 2 | 0 |
| `P2` | 4 | 3 |
| `P3` | 6 | 6 |
| `P4` | 4 | 2 |
| `P5` | 11 | 13 |
| `P6` | 1 | 0 |

## Directory Guide For Results

If you want to browse the outputs systematically, use this order.

### 1. Reconstruction Error CSVs

Folder:

- `data/processed/step7`

Files:

- `swat_P1_anomaly_scores.csv`
- `swat_P2_anomaly_scores.csv`
- `swat_P3_anomaly_scores.csv`
- `swat_P4_anomaly_scores.csv`
- `swat_P5_anomaly_scores.csv`
- `swat_P6_anomaly_scores.csv`

What they contain:

- one row per time window
- one column per process variable for that stage
- reconstruction MSE per feature

### 2. Reconstruction-Based RCA Outputs

Folder:

- `data/processed/step8`

Files:

- `swat_rca_results.csv`
- `anomalies.json`

What they contain:

- stage-level and feature-level root cause assignments
- anomalous sensor lists and scores
- JSON anomaly objects for the Bayesian RCA step

### 3. Bayesian Network Outputs

Folder:

- `data/processed/step11`

Files:

- `BN_P1.json`
- `BN_P2.json`
- `BN_P3.json`
- `BN_P4.json`
- `BN_P5.json`
- `BN_P6.json`

What they contain:

- node lists
- directed edges
- CPT placeholders or stored CPD structure

### 4. Bayesian RCA Outputs

Folder:

- `data/processed/step12`

Files:

- `swat_rca_step12_results.csv`

What it contains:

- anomaly metadata
- original guilty feature from classical RCA
- Bayesian root cause ranking
- propagation paths
- heuristic confidence score

### 5. LLM-Based Path Evaluation Outputs

Folder:

- `data/processed/step13`

Files:

- `propagation_path_llm_evaluation.csv` (and any timestamped variants)

What they contain:

- one row per anomaly and candidate path
- anomaly metadata (stage, phase, timestamp, anomalous sensors, candidate path nodes)
- LLM verdict fields such as `path_valid`, `confidence` (0–1), and a short `reason`

These outputs are produced by step 13 and provide a semantic plausibility filter over the graph-derived candidate paths from step 12.

### 6. Temporal Consistency Evaluation Outputs

Folder:

- `data/processed/step14`

Files:

- `propagation_path_temporal_evaluation.csv` (and any timestamped variants)
- `propagation_temporal_summary.json` (summary of the run and aggregate metrics)

What they contain:

- expanded candidate paths from step 12 (one row per anomaly/path pair)
- per-path temporal labels (for example, `consistent`, `inconsistent`, `insufficient_evidence`)
- numeric temporal scores between 0 and 1
- flags for anomaly-time alignment and time-order support
- interface labels and reasons for stage-boundary sensors
- serialized per-sensor change times used for the temporal reasoning

These outputs are produced by step 14 and quantify how well the path order and onset times line up with the anomaly time.

#### Temporal Label Counts (Multi-Node Paths)

From the current `propagation_path_temporal_evaluation.csv` (considering only paths with `path_length >= 2`). These counts are automatically refreshed by step 15 and exclude single-node paths from both the table and the temporal figures.

<!-- STEP15_TEMPORAL_LABEL_COUNTS_START -->
| Label | Count |
|---|---:|
| `consistent` | 74 |
| `time_order_supported` | 34464 |
| `inconsistent` | 2131 |
| `insufficient_evidence` | 3849 |
<!-- STEP15_TEMPORAL_LABEL_COUNTS_END -->

### 7. Temporal Analysis And Diagnostic Outputs

Folders:

- `data/processed/step15`
- `reports/figures`

Key files:

- `data/processed/step15/step15_complete_multinode_paths.csv`
- `data/processed/step15/step15_null_change_time_sensor_counts.csv`
- `data/processed/step15/step15_temporal_analysis_summary.json`
- `reports/figures/step15_temporal_label_counts.png`
- `reports/figures/step15_stage_temporal_label_distribution.png`
- `reports/figures/step15_temporal_score_histogram.png`
 - `reports/figures/step15_interface_label_counts.png`

What they contain:

- label counts and temporal score distributions for multi-node paths
- stage-wise breakdown of temporal labels
- counts of how often each sensor lacks a detected change time
- a filtered comparison table of multi-node paths with complete timestamps
- a compact JSON summary with key statistics and figure paths
- interface-label distributions summarizing stage-boundary checks

These outputs are produced by step 15 and make the temporal evaluation layer easy to inspect visually.

### 8. Plot Outputs

Folder:

- `reports/figures`

Main files:

- `step8_timeline_P1.png` through `step8_timeline_P6.png`
- `step8_recon_*.png`
- `step10_rca_stage_dist.png`
- `step10_rca_feature_top10.png`
- `step10_rca_P1_features.png`
- `step10_rca_P2_features.png`
- `step10_rca_P3_features.png`
- `step10_rca_P5_features.png`
- `step10_rca_P6_features.png`

### 9. Bayesian Graph Images

Folder:

- `reports/figures/bn_graphs`

Files:

- `BN_all_stages_overview.png`
- `BN_P1_graph.png`
- `BN_P2_graph.png`
- `BN_P3_graph.png`
- `BN_P4_graph.png`
- `BN_P5_graph.png`
- `BN_P6_graph.png`

## Main Result Figures

### 1. Stage Distribution Of Anomalies

This chart summarizes which stage was most frequently selected as the guilty stage by the classical RCA logic.

![Distribution of anomalies by guilty stage](reports/figures/step10_rca_stage_dist.png)

What it shows:

- `P6` dominates the current output
- `P3` and `P2` also contribute heavily
- `P4` is almost absent as the top cause

### 2. Top 10 Global Anomalous Sensors Or Actuators

This chart shows the global count of the most frequent guilty features.

![Top 10 anomalous sensors and actuators](reports/figures/step10_rca_feature_top10.png)

What it shows:

- `FIT601.Pv` dominates the global ranking
- `AIT303.Pv`, `FIT201.Pv`, and `FIT301.Pv` are the next strongest recurring root-cause features

### 3. Stage-Specific RCA Feature Summaries

These plots show which features dominate inside each stage-specific RCA subset.

#### P1

![Top root causes for stage P1](reports/figures/step10_rca_P1_features.png)

Main takeaway:

- P1 is mainly driven by `FIT101.Pv` and `LIT101.Pv`

#### P2

![Top root causes for stage P2](reports/figures/step10_rca_P2_features.png)

Main takeaway:

- `FIT201.Pv` dominates P2 by a large margin
- `AIT201.Pv` appears but far less often

#### P3

![Top root causes for stage P3](reports/figures/step10_rca_P3_features.png)

Main takeaway:

- P3 is dominated by `AIT303.Pv` and `FIT301.Pv`

#### P5

![Top root causes for stage P5](reports/figures/step10_rca_P5_features.png)

Main takeaway:

- P5 is almost entirely driven by `AIT503.Pv`
- `PIT502.Pv` appears only occasionally

#### P6

![Top root causes for stage P6](reports/figures/step10_rca_P6_features.png)

Main takeaway:

- P6 has a single modeled process variable, so `FIT601.Pv` necessarily dominates

## Anomaly Detection Timelines

These figures show the stage-level reconstruction error over time with the normal-derived threshold.

### P1 Timeline

![P1 anomaly timeline](reports/figures/step8_timeline_P1.png)

### P2 Timeline

![P2 anomaly timeline](reports/figures/step8_timeline_P2.png)

### P3 Timeline

![P3 anomaly timeline](reports/figures/step8_timeline_P3.png)

### P4 Timeline

![P4 anomaly timeline](reports/figures/step8_timeline_P4.png)

### P5 Timeline

![P5 anomaly timeline](reports/figures/step8_timeline_P5.png)

### P6 Timeline

![P6 anomaly timeline](reports/figures/step8_timeline_P6.png)

## Reconstruction Comparison Plots

These plots compare actual versus reconstructed signals, with green shading for normal regions and red shading for attack regions.

### P1

![P1 reconstruction for LIT101](reports/figures/step8_recon_P1_LIT101.Pv.png)

![P1 reconstruction for FIT101](reports/figures/step8_recon_P1_FIT101.Pv.png)

### P2

![P2 reconstruction for AIT201](reports/figures/step8_recon_P2_AIT201.Pv.png)

![P2 reconstruction for FIT201](reports/figures/step8_recon_P2_FIT201.Pv.png)

### P3

![P3 reconstruction for AIT301](reports/figures/step8_recon_P3_AIT301.Pv.png)

![P3 reconstruction for AIT302](reports/figures/step8_recon_P3_AIT302.Pv.png)

### P4

![P4 reconstruction for FIT401](reports/figures/step8_recon_P4_FIT401.Pv.png)

![P4 reconstruction for LIT401](reports/figures/step8_recon_P4_LIT401.Pv.png)

### P5

![P5 reconstruction for FIT501](reports/figures/step8_recon_P5_FIT501.Pv.png)

![P5 reconstruction for FIT502](reports/figures/step8_recon_P5_FIT502.Pv.png)

### P6

![P6 reconstruction for FIT601](reports/figures/step8_recon_P6_FIT601.Pv.png)

### Interpretation Of Reconstruction Plots (How To Read)

- black line: actual normalized process value
- cyan line: autoencoder reconstruction
- large separation: high reconstruction error (anomaly evidence)
- red shading: attack windows

## Bayesian Network Visualizations

### All-Stage Overview

This is the fastest way to inspect all learned stage graphs in one figure.

![All stage Bayesian Network overview](reports/figures/bn_graphs/BN_all_stages_overview.png)

### P5 Graph

![P5 Bayesian Network graph](reports/figures/bn_graphs/BN_P5_graph.png)

### P6 Graph

![P6 Bayesian Network graph](reports/figures/bn_graphs/BN_P6_graph.png)

### Other Stage Graphs

#### P1

![P1 Bayesian Network graph](reports/figures/bn_graphs/BN_P1_graph.png)

#### P2

![P2 Bayesian Network graph](reports/figures/bn_graphs/BN_P2_graph.png)

#### P3

![P3 Bayesian Network graph](reports/figures/bn_graphs/BN_P3_graph.png)

#### P4

![P4 Bayesian Network graph](reports/figures/bn_graphs/BN_P4_graph.png)

## Temporal Consistency Figures (Step 15)

### Label Counts

Distribution of temporal labels for multi-node paths.

![Temporal label counts](reports/figures/step15_temporal_label_counts.png)

### Stage-Wise Label Distribution

Stage-wise stacked counts of temporal labels for multi-node paths.

![Stage-wise temporal label distribution](reports/figures/step15_stage_temporal_label_distribution.png)

### Temporal Score Histogram

Histogram of temporal scores for multi-node paths.

![Temporal score histogram](reports/figures/step15_temporal_score_histogram.png)

### Interface Label Counts

Distribution of interface labels (consistent, inconsistent, insufficient_evidence) for multi-node paths.

![Interface label counts](reports/figures/step15_interface_label_counts.png)

## Step 12 Bayesian RCA Output

The Bayesian RCA output file is:

- `data/processed/step12/swat_rca_step12_results.csv`

This file currently has `12306` rows, matching the classical RCA anomaly count.



<!-- STEP15_TEMPORAL_LABEL_COUNTS_START -->
| Label | Count |
|---|---:|
| `consistent` | 74 |
| `time_order_supported` | 34464 |
| `inconsistent` | 2131 |
| `insufficient_evidence` | 3849 |
<!-- STEP15_TEMPORAL_LABEL_COUNTS_END -->

### What Changes Relative To Step 9

Compared with `swat_rca_results.csv`, the step 12 file adds:

- `root_causes`: ranked BN-based root candidates with scores
- `propagation_paths`: graph paths from root candidates to deviating nodes
- `confidence`: a heuristic confidence number

## LLM Path Evaluation And Temporal Consistency Layers

### LLM-Based Path Evaluation (Step 13)

The LLM evaluation CSV in `data/processed/step13` adds, for each anomaly/path pair:

- a boolean verdict on whether the path is a plausible explanation
- a confidence score between 0 and 1
- a short, targeted natural-language reason

In practice, this acts as a semantic filter on the raw graph-derived paths from step 12, down-weighting paths that fail basic coverage or causal-direction checks based on SWaT-specific context.

### Temporal Consistency (Steps 14–15)

The temporal evaluation and analysis layers in `data/processed/step14`, `data/processed/step15`, and the step 15 figures provide:

- per-path scores for how well sensor change times respect the proposed propagation order
- labels that distinguish well-ordered, partially ordered, and inconsistent paths
- diagnostics on which sensors frequently lack reliable change-time estimates

Together with the LLM verdicts, these layers help highlight propagation explanations that are both semantically and temporally plausible.

## Recommended Reading Order For Outputs

If someone new to the repository wants to inspect the results efficiently, this is the best order:

1. start with `reports/figures/step10_rca_stage_dist.png`
2. then inspect `reports/figures/step10_rca_feature_top10.png`
3. then look at the six `step8_timeline_*.png` files
4. then inspect the reconstruction plots for the dominant stages P2, P3, P5, and P6
5. then open `reports/figures/bn_graphs/BN_all_stages_overview.png`
6. then inspect `BN_P5_graph.png` and `BN_P3_graph.png`
7. then compare `data/processed/step8/swat_rca_results.csv` with `data/processed/step12/swat_rca_step12_results.csv`
8. then inspect `data/processed/step13/propagation_path_llm_evaluation.csv`
9. finally, inspect `data/processed/step14/propagation_path_temporal_evaluation.csv` together with the step 15 plots and summary tables

## Practical Summary

The repository provides:

- anomaly score timelines and reconstruction plots
- classical RCA CSVs and summary figures
- Bayesian Network JSONs and graph images
- Bayesian RCA result tables
- LLM-based propagation path evaluations
- temporal consistency scores, plots, and diagnostic tables

For quick browsing, use `reports/figures` for anomaly/RCA/temporal plots and `reports/figures/bn_graphs` for causal graph visualizations.