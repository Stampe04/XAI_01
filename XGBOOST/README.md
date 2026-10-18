# XGBoost diabetes model

This folder trains the tree-based classifier used in the project comparison and explores how its predictions can be explained. The training script fits and evaluates the model; the notebook examines global importance and individual predictions with LIME and SHAP. Read the [root README](../README.md) for the broader XAI goals and interpretation limits.

| Path | Purpose |
| --- | --- |
| `diabetes_xgboost.py` | Data loading, stratified splitting, tuning, training, and holdout evaluation. |
| [`config/`](config/README.md) | Hydra settings for data, model, tuning, and output paths. |
| `xgboost.ipynb` | Feature importance, LIME, SHAP, and local/global explanation comparisons. |
| `artifacts/` | Locally generated model, metrics, and resolved run configuration; ignored by Git. |

The project uses [UV](https://docs.astral.sh/uv/) and the dependency versions in `../uv.lock`.

From `XAI_01`:

```powershell
uv sync
uv run python XGBOOST/diabetes_xgboost.py
```

This trains on the balanced BRFSS binary dataset used by `nn.ipynb`, evaluates a stratified 80/20 holdout, and writes the fitted model, metrics, and resolved run configuration to `XGBOOST/artifacts/`. Those generated files are ignored by Git. The model uses numeric inputs without scaling.

Early stopping is enabled by default. A stratified validation slice **inside the training set** chooses the number of boosting rounds (with patience 30); the final model is then refit on all training rows using that number of trees. Its configured maximum is 800, but a tuned candidate's sampled `n_estimators` can set a lower limit. The test set is used only for final evaluation. Set `early_stopping.enabled=false` to use the fitted model's `n_estimators` value.

`config/config.yaml` records the master random seed (`seed: 42`). The script uses it for splits, randomized search, Optuna sampling, and XGBoost's row/column sampling; Optuna's inner validation folds use that seed plus the fold index. Each run copies the configuration to `artifacts/run_config.yaml` and records the seed in `artifacts/metrics.json`. Override it with `seed=123` to repeat the whole workflow with another seed.

The default command runs an Optuna TPE search with 20 trials. To use stratified randomized search instead:

```powershell
uv run python XGBOOST/diabetes_xgboost.py tuning.method=random
```

Both randomized search and Optuna sample from the ranges under `tuning.search_space` in the Hydra config. Integer bounds include both endpoints; `learning_rate` is sampled continuously on a log scale. See [`config/config.yaml`](config/config.yaml) for the current values. With early stopping enabled, the sampled `n_estimators` becomes the upper limit for that candidate; early stopping can select fewer trees. The best sampled limit also caps the final fit.

To train without hyperparameter tuning:

```powershell
uv run python XGBOOST/diabetes_xgboost.py tuning.enabled=false
```

Optuna's default 20 trials score mean ROC-AUC across three stratified outer folds. Each fold uses a separate inner validation slice for early stopping, keeping its scoring fold unseen until evaluation. After tuning, the script selects the final boosting rounds on a training-only validation slice and refits on all training rows. The Hydra config controls the search ranges, trial count, and early-stopping patience. Both search modes leave the test holdout untouched until final evaluation.

Other settings in `XGBOOST/config/config.yaml` can be changed with Hydra overrides, for example `model.max_depth=3`, `tuning.n_iter=16`, or `tuning.optuna.n_trials=40`.

Open `XGBOOST/xgboost.ipynb` with the Python interpreter in `XAI_01/.venv` and run all cells after training. It reloads the saved model and configuration, recreates the same holdout, and contains the week 1–3 explainability work: gain versus permutation importance, LIME, SHAP, and local/global comparisons. Rerun it after changing the model so its explanations match the fitted artifact. Clear cell outputs before committing the notebook; generated plots are kept locally.
