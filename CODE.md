# CODE

## Purpose Of This Document

This file explains the repository from the code and pipeline perspective.
It documents:

- what each script does
- which files it expects as input
- which files it writes as output
- where generated artifacts are stored
- how each step depends on previous steps
- what is important to know about implementation details and assumptions

This document covers the main pipeline files:

- `notebooks/step1.py`
- `notebooks/step2.py`
- `notebooks/step3.py`
- `notebooks/step4.py`
- `notebooks/step5.py`
- `notebooks/step6.py`
- `notebooks/step7.py`
- `notebooks/step8.py`
- `notebooks/step9.py`
- `notebooks/step10.py`
- `notebooks/step11.py`
- `notebooks/step12.py`
- `notebooks/step13.py`
- `notebooks/step14.py`
- `notebooks/step15.py`

It also includes the interactive RCA chatbot entrypoint:

- `chatbot.py`

It also includes the helper visualization script:

- `notebooks/graph_view.py`

And one compact context helper file used by the chatbot prompt:

- `CONTEXT_COMPACT.md`

## Overall Pipeline Summary

At a high level, the repository implements seven stacked layers of logic:

1. tabular preprocessing of SWaT telemetry
2. time-window creation and stage-wise LSTM anomaly modeling
3. stage-level and feature-level root cause analysis from reconstruction error
4. causal modeling and Bayesian root cause analysis using learned stage graphs
5. local LLM evaluation of candidate anomaly propagation paths
6. temporal consistency evaluation and diagnostics for propagation paths
7. interactive chatbot querying over static step16 outputs with deterministic and optional Groq-backed responses

The code flows forward through the `data/processed` tree. Each later step assumes the required earlier artifacts already exist.

## Execution Order

The intended run order is:

1. `step1.py`
2. `step2.py`
3. `step3.py`
4. `step4.py`
5. `step5.py`
6. `step6.py`
7. `step7.py`
8. `step8.py`
9. `step9.py`
10. `step10.py`
11. `step11.py`
12. `step12.py`
13. `step13.py`
14. `step14.py`
15. `step15.py`

Then optionally:

1. `graph_view.py`
2. `chatbot.py`

In practice, `step1.py` is exploratory and does not create a reusable output file in its current implementation, while `step2.py` through `step15.py` generate artifacts used by later stages.
`chatbot.py` is a serving/application layer that reads already-generated artifacts and does not write pipeline outputs.

## Directory Conventions

The repository stores outputs in a few main locations:

- `data/processed/step2` through `data/processed/step15`: intermediate and final data outputs
- `models/lstm`: saved PyTorch model weights
- `reports/figures`: anomaly and RCA figures
- `reports/figures/bn_graphs`: Bayesian Network graph images

It also contains root-level application/context files:

- `chatbot.py`: Streamlit chatbot app for interactive analysis
- `weaviate_explorer.py`: CLI utility for exploring and exporting RCA results from Weaviate
- `CONTEXT_COMPACT.md`: compact domain/pipeline context used to reduce prompt token usage

One implementation detail matters: some earlier scripts use relative paths like `../data/...`, while later scripts compute `SCRIPT_DIR` and `PROJECT_ROOT` explicitly. That means the early steps are written with the expectation that they are usually run from the `notebooks` directory.

## Step 1: `notebooks/step1.py`

Exploratory time-and-phase inspection: loads the cleaned SWaT table, converts timestamps, defines normal and attack regions, and plots a few variables with shaded periods to establish the timeline used later.

## Step 2: `notebooks/step2.py`

Metadata extraction: scans all columns in the cleaned SWaT table, classifies them into types (Pv, Status, Alarm, State, Other), infers stage P1–P6, and writes a feature metadata table used by all later steps.

## Step 3: `notebooks/step3.py`

Main tabular cleaning: drops alarm columns, normalizes dtypes for process/status/state fields, forward/back-fills missing values, and writes the cleaned base table used for normalization.

## Step 4: `notebooks/step4.py`

Normalization: uses normal-period data to compute mean/std for all Pv columns, applies Z-score normalization, and saves both the normalized table and the scaler stats for later reuse.

## Step 5: `notebooks/step5.py`

Windowing: turns the normalized table into continuous 60-second sliding windows, labels each window by operating phase, splits normal vs non-normal, and saves all arrays plus window timestamps.

## Step 6: `notebooks/step6.py`

Stage splitting: uses metadata to map Pv columns to stages P1–P6, slices the global window tensors into per-stage arrays, and saves both the stage-wise tensors and the `stage_feature_map` used throughout the rest of the pipeline.

## Step 7: `notebooks/step7.py`

Stage-wise LSTM training: for each stage trains an LSTM autoencoder on normal windows, saves best checkpoint weights, runs inference on all windows, and exports per-feature reconstruction-error scores as anomaly evidence.

## Step 8: `notebooks/step8.py`

Inference and plots: reloads trained LSTM autoencoders, recomputes feature-wise errors and stage-level scores, derives high-percentile thresholds, and produces anomaly timelines plus reconstruction comparison plots per stage.

## Step 9: `notebooks/step9.py`

Classical RCA: combines per-stage anomaly scores with thresholds to flag anomalous windows, picks a guilty stage/feature, records top anomalous sensors, and writes both CSV and JSON anomaly descriptions that feed the Bayesian step.

## Step 10: `notebooks/step10.py`

RCA summaries: reads the step 9 RCA CSV and produces distribution plots of anomalous windows by stage and top guilty features globally and per stage.

## Step 11: `notebooks/step11.py`

BN learning: aggregates normal windows per stage, removes near-constant features, discretizes values, learns Bayesian Network structure/CPDs, saves BN JSONs, and pushes corresponding sensor nodes and causal edges into Weaviate.

## Step 12: `notebooks/step12.py`

Bayesian RCA: loads anomaly descriptions and stage BNs, rebuilds/optionally refits models, runs inference to score root-cause candidates, extracts propagation paths, computes a confidence heuristic, and stores results both in Weaviate and as a local CSV.

## Step 13: `notebooks/step13.py`

LLM path evaluation: normalizes candidate propagation paths (from step 12 or a custom CSV) into row-wise records, prompts a local Ollama `qwen2.5:7b` model for JSON judgments, and writes structured plausibility scores and reasons.

## Step 14: `notebooks/step14.py`

Temporal evaluation: combines candidate paths with per-feature anomaly scores and window timestamps, derives change/onset times per sensor, scores temporal order and anomaly-time alignment for each path, checks interface sensors at stage boundaries, and writes a detailed evaluation CSV plus a compact summary JSON.

## Step 15: `notebooks/step15.py`

Temporal diagnostics: ingests the step 14 summary/CSV, focuses on multi-node paths with complete timestamps, generates label-count, stage-distribution, temporal-score, and interface-label plots, and writes comparison tables plus a compact temporal analysis summary.

## Helper Script: `notebooks/graph_view.py`

BN visualization: loads each stage BN JSON, builds a directed graph with `networkx`, and renders per-stage and all-stage overview images under `reports/figures/bn_graphs`.

## Step 17: `chatbot.py`

Interactive RCA chatbot: serves a Streamlit app that reads static step16 outputs, supports deterministic handling for structured questions (for example low/high confidence ranking and stage comparison), and can optionally call Groq for concise narrative responses grounded on selected step16 rows and `CONTEXT_COMPACT.md`.

This step is intentionally non-destructive: it does not retrain models or modify prior pipeline artifacts.

## Context Helper: `CONTEXT_COMPACT.md`

Prompt grounding helper: stores a compact summary of project/domain context (derived from `INFO.md` and `CODE.md`) so LLM prompts stay contextually accurate with lower token cost.

## Input And Output Dependency Chain

This section summarizes the artifact chain in compact form.

### Base Input

- `data/processed/swat_cleaned.csv`

### Metadata Layer

- `step2.py` writes `data/processed/step2/swat_feature_metadata.csv`

### Cleaned And Normalized Tables

- `step3.py` writes `data/processed/step3/swat_step3_clean.csv`
- `step4.py` writes `data/processed/step4/swat_step4_normalized.csv`
- `step4.py` writes `data/processed/step4/swat_pv_scaler.pkl`

### Time-Series Windows

- `step5.py` writes all window arrays into `data/processed/step5`

### Stage-Specific Model Inputs

- `step6.py` writes per-stage arrays and `stage_feature_map.json` into `data/processed/step6`

### Learned Models And Feature Errors

- `step7.py` writes `.pt` weights into `models/lstm`
- `step7.py` writes per-stage feature error CSVs into `data/processed/step7`

### Visual Detection Outputs

- `step8.py` writes plots into `reports/figures`
- `step8.py` also refreshes the same anomaly score CSVs in `data/processed/step7`

### Classical RCA Outputs

- `step9.py` writes RCA CSV and anomaly JSON into `data/processed/step8`
- `step10.py` writes RCA summary figures into `reports/figures`

### Bayesian Causal Outputs

- `step11.py` writes BN JSON files into `data/processed/step11`
- `graph_view.py` writes BN visualizations into `reports/figures/bn_graphs`
- `step12.py` writes Bayesian RCA results into `data/processed/step12`

### LLM And Temporal Path Evaluation Outputs

- `step13.py` writes LLM-based path plausibility scores into `data/processed/step13`
- `step14.py` writes temporal consistency evaluations and a summary JSON into `data/processed/step14`
- `step15.py` writes temporal analysis tables into `data/processed/step15` and plots into `reports/figures`

### Chatbot Serving Input

- `chatbot.py` reads `data/processed/step16/llm_explanations.csv` as its primary static anomaly-explanation source
- `chatbot.py` also reads `CONTEXT_COMPACT.md` for compact repository/domain grounding

## Key Implementation Quirks To Be Aware Of

These are worth knowing if you continue developing this codebase.

### 1. Early Path Handling Is Less Robust Than Later Path Handling

Steps 1, 3, 4, 7, and 10 rely heavily on relative paths such as `../data/...` and `../reports/...`.
Steps 5, 6, 8, 11, 12, and `graph_view.py` are more robust because they compute absolute project-relative paths from `__file__`.

### 2. Step 1 Is More Diagnostic Than Transformational

Despite being listed as the first pipeline step, it does not currently create an artifact used later in the codebase.

### 3. Step 8 Writes Into The Step 7 Output Directory

This is not wrong, but it is easy to misunderstand. Step 8 recomputes and saves feature-wise anomaly scores to `data/processed/step7`.

### 4. Step 9 Feeds Step 12

Even though step 12 is Bayesian RCA, it still depends on the anomaly candidate formatting prepared by step 9.

### 5. The Bayesian Pipeline Uses Only Normal Data For Structure Learning

This means the graph represents learned normal dependencies, and anomalies are interpreted against that learned normal structure.

## Hosted Chatbot

You can try the interactive RCA chatbot online:

- [chatbot.py Streamlit app](https://rfc-chatbot.streamlit.app/)

_Note: If the app has been idle, it may take up to 60 seconds to wake up._

## Practical Reading Order For Developers

If you want to understand the code efficiently, read files in this order:

1. `step2.py`: learn how columns are typed and staged
2. `step5.py`: learn how windows are created and labeled
3. `step6.py`: learn how the global tensor becomes stage-specific data
4. `step7.py`: understand the autoencoder architecture and training logic
5. `step9.py`: understand reconstruction-error RCA
6. `step11.py`: understand BN learning
7. `step12.py`: understand Bayesian RCA

Then read `step8.py`, `step10.py`, and `graph_view.py` for visualization and reporting layers.

## Practical Summary

The codebase is organized as a staged artifact pipeline.

- steps 1 to 4 prepare a clean, normalized plant table
- steps 5 and 6 convert that table into stage-wise time windows
- steps 7 and 8 train sequence models and score anomalies
- steps 9 and 10 summarize reconstruction-based root causes
- steps 11 and 12 add Bayesian causal structure and graph-based RCA
- steps 13 to 15 evaluate propagation paths with an LLM, check temporal consistency, and visualize temporal diagnostics

Every important output in the repository is stored under `data/processed`, `models/lstm`, or `reports/figures`.
The root-level chatbot (`chatbot.py`) is a consumer-facing interface over those generated artifacts.

## Utility Script: `weaviate_explorer.py`

This command-line tool allows interactive exploration and export of RCA results stored in a Weaviate instance. Features include:
- Filtering by stage, root cause, or timestamp
- Exporting results to CSV
- Interactive shell for browsing RCA objects
- Useful for advanced analysis or integration with external tools

## Requirements Files

- `requirements.txt`: Main dependencies for running the pipeline and chatbot interface.
- `requirements-models.txt`: Additional dependencies for model training, advanced analytics, or optional features.
