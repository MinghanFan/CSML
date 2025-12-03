# CSML

Parse CS2 demos into clean tables, engineer features, and train a Bayesian Network and LightGBM round predictor.

## Data Pipeline
- **Parse/Extract:** `scripts/remote_pipeline.py` streams remote/local demos; progress logged in `clean_dataset/processed_remote_demos.txt`. Use `--output-folder/--output-prefix` to isolate runs; merge with `scripts/merge_datasets.py`.
- **Tables:** `clean_dataset/` holds `matches.csv`, `rounds.csv`, `round_players.csv`, `players.csv` (see `scripts/data_schema.py`). `scripts/verify_dataset.py` checks integrity.

## Bayesian Network
- **Prep:** `scripts/bn_data_preparation.py` builds round-level features (economy buckets, map bias, score pressure, momentum-like stats), discretizes them, and writes `clean_dataset/bn_analysis/rounds_discretized.csv` plus plots (`feature_distributions.png`, `feature_dependencies.png`, `scenario_analysis.*`). Verbose logs can be enabled via `VERBOSE=True`.
- **Train:** `scripts/bn_model.py` fits a discrete BN with explicit state spaces and 5-fold GroupKFold CV. Outputs: CPDs (`bn_analysis/bn_cpd_tables/`), CV results (`bn_cv_results.csv`), metrics (`bn_metrics.json`), calibration curve (`bn_calibration.png`), inference examples (`bn_inference_examples.json`). pgmpy INFO logs are suppressed.
- **Viz:** `scripts/bn_viz.py` renders the BN structure (`bn_structure.png`), BN vs LightGBM comparison (`bn_vs_lightgbm.png`), and sampled inference bars (`bn_inference_examples.png`).

## LightGBM Baseline
- **Train:** `scripts/lgb_model.py` trains a calibrated LightGBM classifier. Outputs: `lgb_model.txt` (booster), `lgb_model.pkl` (calibrator + metadata), `lgb_feature_importance.csv`, `lgb_cv_scores.csv`, `lgb_metrics.json`.

## Clustering
- **Player styles:** `scripts/cluster.py` (feature engineering → PCA → KMeans). Artifacts: `clean_dataset/match_players_with_clusters.csv`, `player_cluster_profiles.csv`, `player_clusters_pca.png`.
- **Evaluation:** `scripts/cluster_evaluation.py` produces silhouette/elbow plots in `cluster_eval/`.

## Process
1) Parse demos: `python scripts/remote_pipeline.py --local-folder /path/to/demos --output-folder clean_dataset` (or `--root-url/--manifest`).
2) Prep BN data: `python scripts/bn_data_preparation.py` → see `clean_dataset/bn_analysis/`.
3) Train BN: `python scripts/bn_model.py` → check `bn_metrics.json`, `bn_calibration.png`.
4) Train LGB: `python scripts/lgb_model.py` → check `lgb_metrics.json`, `lgb_feature_importance.csv`.
