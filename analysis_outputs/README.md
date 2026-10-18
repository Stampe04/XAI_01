# Local quantitative analysis outputs

`../run_quantitative_analysis.py` generates this folder after the XGBoost model has been trained. Run it from the repository root:

```powershell
uv run python run_quantitative_analysis.py
```

The script writes model validation and explanation figures as PNG and SVG pairs, plus CSV tables for holdout predictions, error slices, Quantus scores, feature flipping, stability, agreement, and the evaluation profile. It also writes `results.json`, `attributions.npz`, and a local neural-network checkpoint (`nn_state.pt`). Open [`quantitative_analysis.ipynb`](../quantitative_analysis.ipynb) after generation; it reads these files rather than embedding the figures.

Only this README is tracked. Git ignores every other file in `analysis_outputs/`, including generated images, CSVs, model files, and Matplotlib's cache. Keep or remove those results locally as needed. See the [project README](../README.md) for what the metrics can and cannot tell us.
