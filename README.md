# Explainable AI for diabetes-status classifiers

This project studies how two models, XGBoost and a small neural network, classify reported prediabetes or diabetes status in the balanced 2015 BRFSS survey. These models can be treated as black boxes: they produce predictions without making their reasoning obvious. Explainable AI (XAI) helps us inspect which survey answers influence a model's predictions, compare patterns across people and models, and identify explanations that deserve closer scrutiny.

## How XAI is used here

- **Global inspection:** feature importance and summaries of local attributions show which inputs the models tend to use. They can reveal surprising dependencies, but averages can hide differences between people.
- **Local inspection:** LIME approximates a model near one survey record; SHAP assigns contributions relative to a reference distribution. These describe a model's score for that record, not what would happen if the person changed an answer.
- **Checking explanations:** Quantus faithfulness correlation, feature flipping, and MaxSensitivity test how explanations relate to model behavior under input changes. Sparseness, complexity, and agreement describe their form and similarity. No single metric establishes that an explanation is correct.

The data are cross-sectional and balanced for analysis. A score is about **reported status in this dataset**, not an individual's future risk. Attributions and perturbations do not establish causes or medical advice.

## Repository guide

| Path | Contents |
| --- | --- |
| [`data/`](data/README.md) | Where the local BRFSS CSV files go; the data files are ignored by Git. |
| [`data.ipynb`](data.ipynb) | Exploratory KaggleHub download notebook. |
| [`nn.ipynb`](nn.ipynb) | Neural-network training and early LIME, SHAP, and validation experiments. It currently contains a machine-specific CSV path; adjust that path before running it elsewhere. |
| [`XGBOOST/`](XGBOOST/README.md) | Configured XGBoost training pipeline and its explanation notebook. |
| [`run_quantitative_analysis.py`](run_quantitative_analysis.py) | Reproducible comparison on a shared holdout, including model metrics and Quantus explanation checks. |
| [`quantitative_analysis.ipynb`](quantitative_analysis.ipynb) | Reads locally generated tables and figures and presents the comparison. |
| [`analysis_outputs/`](analysis_outputs/README.md) | Local figures, tables, model checkpoint, and machine-readable results; generated files are ignored by Git. |
| `pyproject.toml`, `uv.lock` | Python dependencies and their lockfile. |

## Run the comparison

Use Python 3.12 or newer and [uv](https://docs.astral.sh/uv/). Place `diabetes_binary_5050split_health_indicators_BRFSS2015.csv` at the path described in [`data/README.md`](data/README.md). From the repository root:

```powershell
uv sync
uv run python XGBOOST/diabetes_xgboost.py
uv run python run_quantitative_analysis.py
```

The first script trains and saves the XGBoost model locally. The second trains or reloads the neural network, evaluates both models on the same stratified holdout, and writes results to `analysis_outputs/`. Open `quantitative_analysis.ipynb` after it finishes. The XGBoost and output-folder READMEs describe the methods and generated files in more detail.

## Keeping the guides current

When changing a workflow, dataset path, model, explanation method, or repository layout, update this overview and the README in the affected folder in the same change. Generated images, CSVs, and model files remain local. Clear notebook cell outputs before committing so embedded plots are not pushed; commit the code and documentation needed to recreate them.
