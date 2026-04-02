# SWaT RCA Project Context (Compact)

Goal:

- Detect SWaT process anomalies and explain likely causes/propagation.

Plant stages:

- P1 Raw Water Intake
- P2 Pre-treatment
- P3 Ultra-Filtration
- P4 De-Chlorination
- P5 Reverse Osmosis
- P6 Disposition

Signal conventions:

- .Pv = continuous process value (main modeling signals)
- .Status = actuator/equipment state
- .Alarm = alarm flags (dropped before modeling)

Time labels (core timeline):

- normal: before 2019-12-06 10:20:00
- cyber_attack: 2019-12-06 10:30:00 to 2019-12-06 11:20:00
- physical_attack: 2019-12-06 12:30:00 to 2019-12-06 13:25:00
- post_attack: at or after 2019-12-06 13:30:00

Pipeline summary:

1) Preprocess + metadata + normalization + windowing (steps 1-6)
2) Stage-wise LSTM autoencoder anomaly scoring (steps 7-8)
3) Reconstruction-error RCA (steps 9-10)
4) Bayesian Network learning + Bayesian RCA (steps 11-12)
5) LLM plausibility scoring of propagation paths (step 13)
6) Temporal consistency evaluation + diagnostics (steps 14-15)

Important modeling assumptions:

- LSTM and BN learning use normal data behavior as reference.
- chatbot chatbot uses static outputs from step16 CSV for anomaly answers.

Key data artifacts:

- base table: data/processed/swat_cleaned.csv
- step16 chat source: data/processed/step16/llm_explanations.csv
