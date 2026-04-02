# INFO

## Purpose Of This Document

This file describes the SWaT data used in this repository:

- what the SWaT dataset is
- what the six plant stages mean here
- how variables are named and typed
- which variables belong to each stage
- how raw telemetry becomes stage-wise model inputs

## What SWaT Is

SWaT (Secure Water Treatment) is a cyber-physical water treatment testbed. The dataset contains time-series measurements from sensors and actuators across multiple treatment stages.

This repository treats SWaT as a six-stage industrial process:

1. `P1`: Raw Water Intake
2. `P2`: Pre-treatment
3. `P3`: Ultra-Filtration
4. `P4`: De-Chlorination
5. `P5`: Reverse Osmosis
6. `P6`: Disposition

The pipeline learns normal behavior from stage-wise signals, detects deviations, and performs RCA using reconstruction error, Bayesian Networks, LLM scoring, and temporal checks.

## What Data Exists In This Repository

The repository currently contains two broad forms of data:

1. Source-like cleaned plant telemetry in `data/processed/swat_cleaned.csv`
2. Derived artifacts under `data/processed/step2` through `data/processed/step12`

The file `data/processed/swat_cleaned.csv` is the starting table used by the pipeline. Although it sits under `processed`, it acts as the main base table for all later steps.

From that table, the pipeline produces:

- metadata about columns and stages
- cleaned and typed tabular data
- normalized process variables
- sliding windows for sequence modeling
- per-stage arrays
- stage-specific LSTM autoencoder scores
- anomaly summaries
- Bayesian Network JSON files
- Bayesian root cause analysis results

## Core Time Structure Used By The Pipeline

The pipeline divides the timeline into operating regions based on timestamps:

- `normal`: before `2019-12-06 10:20:00`
- `cyber_attack`: overlapping `2019-12-06 10:30:00` to `2019-12-06 11:20:00`
- `physical_attack`: overlapping `2019-12-06 12:30:00` to `2019-12-06 13:25:00`
- `post_attack`: at or after `2019-12-06 13:30:00`
- `neutral`: windows that do not fall into the above labels

Notes:

1. Only normal data is used to train LSTM autoencoders and fit Bayesian Networks.
2. Labels are window-based: a 60-second window is anomalous if it overlaps attack spans.

## How The Repository Interprets Variable Names

The naming system follows standard industrial instrument tag patterns.

### Suffixes

- `.Pv`: Process value. This is a numeric measurement used as a continuous signal.
- `.Status`: A discrete actuator or equipment state, usually on or off, open or closed, or running state.
- `.Alarm`: Alarm or alert flag. These are dropped before modeling in this project.
- `P*_STATE`: A stage-level state variable for the corresponding process stage.

### Common Prefixes In This Repository

- `FIT`: Flow Indicator Transmitter
- `LIT`: Level Indicator Transmitter
- `AIT`: Analyzer Indicator Transmitter
- `PIT`: Pressure Indicator Transmitter
- `DPIT`: Differential Pressure Indicator Transmitter
- `MV`: Motorized Valve
- `P`: Pump
- `UV`: UV unit or UV-related actuator
- `LS`, `LSH`, `LSL`, `PSH`, `PSL`, `DPSH`: alarm-oriented limit or pressure switch tags

### How Stage Is Extracted

The pipeline maps tags to stages by using the numeric block inside the tag name.

Examples:

- `101` block belongs to `P1`
- `201` to `299` belong to `P2`
- `301` to `399` belong to `P3`
- `401` to `499` belong to `P4`
- `501` to `599` belong to `P5`
- `601` to `699` belong to `P6`

This is implemented in the metadata extraction step and is the basis for later stage-wise splitting.

## Variable Types Used By The Models

Not every variable is treated equally by the pipeline.

### Variables Kept For LSTM Modeling

Only `.Pv` variables are used for the stage-wise LSTM autoencoders.

That means the anomaly model learns from continuous sensor measurements, not from actuator statuses or alarms.

### Variables Dropped Before Modeling

All `.Alarm` columns are removed in preprocessing.

This is an important design choice. It means the anomaly pipeline tries to detect abnormal behavior from process dynamics rather than from explicit alarm flags that may already indicate failure.

### Variables Present But Not Used In The Stage LSTM Inputs

- `.Status` variables remain in the cleaned table after preprocessing, but they are not included in the stage-wise arrays used by the LSTM autoencoders.
- `P*_STATE` columns are kept in the cleaned table but also excluded from the final per-stage `.Pv` model inputs.

## Stage Overview

The project uses the following process interpretation, taken from the repository README and the metadata mapping logic.

### P1: Raw Water Intake

#### P1 Variables Present In Metadata

Process state:

- `P1_STATE`

Process values:

- `LIT101.Pv`: level-related measurement in stage P1
- `FIT101.Pv`: flow-related measurement in stage P1

Statuses:

- `MV101.Status`: motorized valve state
- `P101.Status`: pump state
- `P102.Status`: pump state

#### P1 Variables Used By The Stage Model

- `LIT101.Pv`
- `FIT101.Pv`

### P2: Pre-treatment

#### P2 Variables Present In Metadata

Process state:

- `P2_STATE`

Process values:

- `FIT201.Pv`: flow-related measurement
- `AIT201.Pv`: analyzer measurement
- `AIT202.Pv`: analyzer measurement
- `AIT203.Pv`: analyzer measurement

Statuses:

- `MV201.Status`
- `P201.Status`
- `P202.Status`
- `P203.Status`
- `P204.Status`
- `P205.Status`
- `P206.Status`
- `P207.Status`
- `P208.Status`

Alarms:

- `LS201.Alarm`
- `LS202.Alarm`
- `LSL203.Alarm`
- `LSLL203.Alarm`

#### P2 Variables Used By The Stage Model

- `FIT201.Pv`
- `AIT201.Pv`
- `AIT202.Pv`
- `AIT203.Pv`

### P3: Ultra-Filtration

#### P3 Variables Present In Metadata

Process state:

- `P3_STATE`

Process values:

- `AIT301.Pv`
- `AIT302.Pv`
- `AIT303.Pv`
- `LIT301.Pv`
- `FIT301.Pv`
- `DPIT301.Pv`

Statuses:

- `MV301.Status`
- `MV302.Status`
- `MV303.Status`
- `MV304.Status`
- `P301.Status`
- `P302.Status`

Alarms:

- `PSH301.Alarm`
- `DPSH301.Alarm`

#### P3 Variables Used By The Stage Model

- `AIT301.Pv`
- `AIT302.Pv`
- `AIT303.Pv`
- `LIT301.Pv`
- `FIT301.Pv`
- `DPIT301.Pv`

### P4: De-Chlorination

#### P4 Variables Present In Metadata

Process state:

- `P4_STATE`

Process values:

- `LIT401.Pv`
- `FIT401.Pv`
- `AIT401.Pv`
- `AIT402.Pv`

Statuses:

- `P401.Status`
- `P402.Status`
- `P403.Status`
- `P404.Status`
- `UV401.Status`

Alarms:

- `LS401.Alarm`

#### P4 Variables Used By The Stage Model

- `LIT401.Pv`
- `FIT401.Pv`
- `AIT401.Pv`
- `AIT402.Pv`

### P5: Reverse Osmosis

#### P5 Variables Present In Metadata

Process state:

- `P5_STATE`

Process values:

- `FIT501.Pv`
- `FIT502.Pv`
- `FIT503.Pv`
- `FIT504.Pv`
- `AIT501.Pv`
- `AIT502.Pv`
- `AIT503.Pv`
- `AIT504.Pv`
- `PIT501.Pv`
- `PIT502.Pv`
- `PIT503.Pv`

Statuses:

- `P501.Status`
- `P502.Status`
- `MV501.Status`
- `MV502.Status`
- `MV503.Status`
- `MV504.Status`

Alarms:

- `PSH501.Alarm`
- `PSL501.Alarm`

#### P5 Variables Used By The Stage Model

- `FIT501.Pv`
- `FIT502.Pv`
- `FIT503.Pv`
- `FIT504.Pv`
- `AIT501.Pv`
- `AIT502.Pv`
- `AIT503.Pv`
- `AIT504.Pv`
- `PIT501.Pv`
- `PIT502.Pv`
- `PIT503.Pv`

### P6: Disposition

#### P6 Variables Present In Metadata

Process state:

- `P6_STATE`

Process values:

- `FIT601.Pv`

Statuses:

- `P601.Status`
- `P602.Status`
- `P603.Status`

Alarms:

- `LSH601.Alarm`
- `LSL601.Alarm`
- `LSH602.Alarm`
- `LSL602.Alarm`
- `LSH603.Alarm`
- `LSL603.Alarm`

#### P6 Variables Used By The Stage Model

- `FIT601.Pv`

## Stage-Wise Process Variable Counts Used By The Model

The final stage feature map used by the LSTM and Bayesian steps is:

| Stage | Count | Process Variables Used |
|---|---:|---|
| `P1` | 2 | `LIT101.Pv`, `FIT101.Pv` |
| `P2` | 4 | `FIT201.Pv`, `AIT201.Pv`, `AIT202.Pv`, `AIT203.Pv` |
| `P3` | 6 | `AIT301.Pv`, `AIT302.Pv`, `AIT303.Pv`, `LIT301.Pv`, `FIT301.Pv`, `DPIT301.Pv` |
| `P4` | 4 | `LIT401.Pv`, `FIT401.Pv`, `AIT401.Pv`, `AIT402.Pv` |
| `P5` | 11 | `FIT501.Pv`, `FIT502.Pv`, `FIT503.Pv`, `FIT504.Pv`, `AIT501.Pv`, `AIT502.Pv`, `AIT503.Pv`, `AIT504.Pv`, `PIT501.Pv`, `PIT502.Pv`, `PIT503.Pv` |
| `P6` | 1 | `FIT601.Pv` |

## What Each Data Processing Step Does To The Data

This section explains the data transformation path, without going into full code details.

### Base Table

Input file:

- `data/processed/swat_cleaned.csv`

Contains timestamped plant measurements and status or alarm columns.

### Metadata Layer

Generated file:

- `data/processed/step2/swat_feature_metadata.csv`

Adds semantic labels to each column:

- variable type
- process stage

### Cleaned Table

Generated file:

- `data/processed/step3/swat_step3_clean.csv`

This version:

- removes alarm columns
- converts `.Pv` values to floating point
- converts `.Status` and `P*_STATE` columns to integer-like types
- fills missing values using forward fill then backward fill

### Normalized Table

Generated files:

- `data/processed/step4/swat_step4_normalized.csv`
- `data/processed/step4/swat_pv_scaler.pkl`

Only `.Pv` columns are normalized using statistics computed from the normal time region.

This prevents attack-period behavior from leaking into the normalization baseline.

### Windowed Data

Generated files under `data/processed/step5`:

- `X_all_windows.npy`
- `X_train_windows.npy`
- `window_phase.npy`
- `window_start_times.npy`
- `window_end_times.npy`

The model sees data as 60-second windows with stride 1. Windows are kept only if timestamps are continuous at 1-second sampling.

### Stage-Wise Arrays

Generated files under `data/processed/step6`:

- `X_all_P1.npy` through `X_all_P6.npy`
- `X_train_P1.npy` through `X_train_P6.npy`
- `stage_feature_map.json`

This is where the single wide plant table becomes six separate stage-specific multivariate sequences.

## Important Modeling Assumptions Embedded In The Data Flow

Several assumptions are built directly into the dataset preparation.

### Assumption 1: Continuous Process Values Are The Main Signals

The anomaly models use only `.Pv` columns.
So the system is oriented around process dynamics rather than explicit control logic state.

### Assumption 2: Normal Behavior Is Learned From A Fixed Early Period

The repository assumes that data before `2019-12-06 10:20:00` is a reliable baseline of normal operation.

### Assumption 3: Stage Boundaries Are Encoded In Tag Numbering

The stage split depends on tag blocks like `101`, `201`, `301`, and so on. This is consistent with the dataset naming convention used in the project.

### Assumption 4: A Window Is The Fundamental Unit Of Detection

The anomaly detector does not classify individual rows independently. It scores 60-second temporal context.

### Assumption 5: Bayesian Causal Learning Uses Aggregated Normal Windows

The Bayesian Network steps later compress each normal window into a single aggregated row, then optionally discretize those values before causal structure learning.

## How To Read A Variable Quickly

Examples:

- `FIT101.Pv`: a continuous flow measurement in stage P1
- `AIT203.Pv`: a continuous analyzer measurement in stage P2
- `MV301.Status`: a discrete motorized valve status in stage P3
- `P5_STATE`: a stage-level state field for stage P5
- `PSH501.Alarm`: a pressure-related alarm field in stage P5

## Data Files Most Important For Understanding The Repository

If you want to inspect the data manually, the most important files are:

- `data/processed/swat_cleaned.csv`: the base time-series table used throughout the project
- `data/processed/step2/swat_feature_metadata.csv`: the authoritative column-to-stage and column-to-type map
- `data/processed/step4/swat_step4_normalized.csv`: the normalized table used before sequence creation
- `data/processed/step5/window_phase.npy`: the phase label assigned to each 60-second window
- `data/processed/step6/stage_feature_map.json`: the exact `.Pv` variables used per stage

## Relation Between Data And Later Results

The structure of the data strongly shapes the later outputs.

- Small stages like `P1` and `P6` produce simple models and limited causal graphs.
- Rich stages like `P3` and `P5` can generate larger anomaly responses and denser Bayesian graphs.
- Any stage with more informative continuous sensors will usually contribute more strongly to stage-level anomaly ranking.

This is why understanding the stage variable inventory is not optional. It directly determines what the models can and cannot learn.

## Practical Summary

At a high level, this repository turns SWaT into a stage-wise industrial time-series problem:

1. start from a timestamped plant telemetry table
2. classify every variable by type and stage
3. keep continuous process values for modeling
4. normalize using normal-only statistics
5. convert the plant history into 60-second windows
6. split those windows into six process stages
7. use those stage-wise signals for anomaly detection and causal analysis

That is the data foundation for everything else in the project.