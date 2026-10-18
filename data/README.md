# Survey data

This folder holds local copies of the 2015 BRFSS diabetes health indicators CSV files. The exploratory [`data.ipynb`](../data.ipynb) uses KaggleHub dataset ID `alexteboul/diabetes-health-indicators-dataset` to obtain them. The CSVs are ignored by Git; download them locally before running the models.

The XGBoost pipeline and quantitative comparison require this file:

```text
data/diabetes-health-indicators-dataset.zip/diabetes_binary_5050split_health_indicators_BRFSS2015.csv
```

Despite its `.zip` suffix, `diabetes-health-indicators-dataset.zip` is a **directory** containing extracted CSV files in this repository layout. The required file is the balanced binary version; the other CSV variants are not used by the main comparison. See the [dataset-folder guide](diabetes-health-indicators-dataset.zip/README.md) for the expected filenames.

Keep the data local. The target `Diabetes_binary` records reported prediabetes or diabetes status in this cross-sectional survey; it is not a future diagnosis label. The scripts create a stratified 80/20 split with seed 42. See the [project README](../README.md) for the analysis workflow.
