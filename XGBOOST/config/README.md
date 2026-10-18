# XGBoost configuration

[`config.yaml`](config.yaml) is the Hydra configuration read by `../diabetes_xgboost.py`. Paths in it are resolved relative to the `XGBOOST` folder, so the training command works from the repository root.

| Section | Controls |
| --- | --- |
| `seed`, `n_jobs` | Reproducibility and training parallelism. |
| `data` | Local CSV path, target column, holdout fraction, and optional row limit. |
| `model` | Base XGBoost classifier settings. |
| `early_stopping` | Training-only validation slice, patience, and maximum boosting rounds. |
| `tuning` | Optuna or randomized search, cross-validation, and shared parameter ranges. |
| `output` | Local artifact directory, ignored by Git. |

From the repository root, run `uv run python XGBOOST/diabetes_xgboost.py`. Hydra overrides can be appended, for example `seed=123`, `tuning.enabled=false`, or `tuning.method=random`. The script saves the resolved settings to `XGBOOST/artifacts/run_config.yaml` with the fitted model and metrics. When these settings change, update the [XGBoost guide](../README.md) and the [project overview](../../README.md) if their descriptions change.
