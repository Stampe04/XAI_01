# Extracted diabetes health indicators files

Place the downloaded CSVs directly in this directory. The folder name ends in `.zip` because of the original download workflow, but the code reads the extracted CSV files, not a ZIP archive.

| CSV | Use in this repository |
| --- | --- |
| `diabetes_binary_5050split_health_indicators_BRFSS2015.csv` | Balanced binary data used by the XGBoost pipeline and quantitative comparison. Required. |
| `diabetes_binary_health_indicators_BRFSS2015.csv` | Unbalanced binary variant; not used by the main comparison. |
| `diabetes_012_health_indicators_BRFSS2015.csv` | Three-class variant; not used by the main comparison. |

The CSV files are ignored by Git. See [`data/README.md`](../README.md) for acquisition and the [project README](../../README.md) for how they are analyzed.
