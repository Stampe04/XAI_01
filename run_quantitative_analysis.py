"""Reproduce the XGBoost/NN comparison and evaluate their explanations.

Run from XAI_01 with ``.venv/Scripts/python run_quantitative_analysis.py``.
The saved XGBoost artefact must be from the checked-in training pipeline.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

_mpl_cache = Path(__file__).resolve().parent / "analysis_outputs" / "mpl_cache"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))

import joblib
import lime.lime_tabular
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quantus
import shap
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis_outputs"
DATA = ROOT / "data/diabetes-health-indicators-dataset.zip/diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
SEED = 42
N_EXPLAIN = 40
N_BACKGROUND = 64
plt.rcParams.update({
    "figure.dpi": 145, "savefig.dpi": 320, "font.size": 10,
    "font.family": "DejaVu Sans", "svg.fonttype": "none",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.labelcolor": "#243347",
    "text.color": "#243347", "xtick.color": "#45566b", "ytick.color": "#45566b",
})


class SimpleNN(nn.Module):
    """Exactly the architecture and optimizer used in nn.ipynb."""

    def __init__(self, input_size: int):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_size, 8), nn.ReLU(), nn.Linear(8, 1))

    def forward(self, x):
        return self.network(x)


class PositiveProbability(nn.Module):
    """The positive-class output used by every explanation quality test."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return torch.sigmoid(self.model(x))


class ProbabilityWrapper(nn.Module):
    """Expose the fitted model's actual two-class probabilities to Quantus."""

    def __init__(self, model, kind: str):
        super().__init__()
        self.model = model if kind == "NN" else None
        self.tree = model if kind == "XGBoost" else None
        self.kind = kind
        self.anchor = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        if self.kind == "NN":
            p = torch.sigmoid(self.model(x)).reshape(-1, 1)
            return torch.cat((1 - p, p), dim=1)
        p = self.tree.predict_proba(x.detach().cpu().numpy())
        return torch.as_tensor(p, dtype=torch.float32, device=x.device)


def savefig(name: str):
    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    svg_path = OUT / Path(name).with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    # Matplotlib pads SVG path coordinates with spaces before line breaks.
    svg = svg_path.read_text(encoding="utf-8")
    svg_path.write_text("\n".join(line.rstrip() for line in svg.splitlines()) + "\n",
                        encoding="utf-8", newline="\n")
    plt.close()


def model_metrics(y, p):
    pred = (p >= .5).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return {
        "n": len(y), "accuracy": float(accuracy_score(y, pred)),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "sensitivity": float(cm[1, 1] / max(1, cm[1].sum())),
        "specificity": float(cm[0, 0] / max(1, cm[0].sum())),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "confusion_matrix": cm.tolist(),
    }


def paired_auc_interval(y, first, second, seed=42, repeats=1000):
    """Paired, label-stratified bootstrap on the untouched holdout."""
    rng = np.random.default_rng(seed)
    negative = np.flatnonzero(y == 0)
    positive = np.flatnonzero(y == 1)
    differences = []
    for _ in range(repeats):
        idx = np.r_[rng.choice(negative, len(negative), replace=True),
                    rng.choice(positive, len(positive), replace=True)]
        differences.append(roc_auc_score(y[idx], first[idx]) -
                           roc_auc_score(y[idx], second[idx]))
    return {"difference": float(roc_auc_score(y, first) - roc_auc_score(y, second)),
            "ci_95": np.quantile(differences, [.025, .975]).tolist()}


def train_nn(X_train, y_train):
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    model = SimpleNN(X_train.shape[1])
    checkpoint = OUT / "nn_state.pt"
    loss_file = OUT / "nn_training_loss.csv"
    if checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        model.eval()
        return model, pd.read_csv(loss_file).loss.tolist() if loss_file.exists() else []
    loader = DataLoader(
        TensorDataset(torch.as_tensor(X_train, dtype=torch.float32),
                      torch.as_tensor(y_train[:, None], dtype=torch.float32)),
        batch_size=256, shuffle=True, generator=torch.Generator().manual_seed(SEED),
    )
    opt = torch.optim.Adam(model.parameters(), lr=.001)
    loss_fn = nn.BCEWithLogitsLoss()
    losses = []
    for epoch in range(100):
        model.train()
        epoch_loss = 0.0
        for x, y in loader:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(loader))
        if (epoch + 1) % 20 == 0:
            print(f"NN epoch {epoch + 1}/100 loss={losses[-1]:.4f}", flush=True)
    model.eval()
    torch.save(model.state_dict(), checkpoint)
    pd.DataFrame({"epoch": np.arange(1, 101), "loss": losses}).to_csv(loss_file, index=False)
    return model, losses


def predict_nn(model, X):
    with torch.no_grad():
        return torch.sigmoid(model(torch.as_tensor(X, dtype=torch.float32))).numpy().ravel()


def slices_table(X, y, probs):
    rows = []
    specs = {
        "Age 1–6": X.Age <= 6, "Age 7–10": X.Age.between(7, 10),
        "Age 11–13": X.Age >= 11, "BMI <25": X.BMI < 25,
        "BMI 25–29": X.BMI.between(25, 29), "BMI ≥30": X.BMI >= 30,
        "HighBP=0": X.HighBP == 0, "HighBP=1": X.HighBP == 1,
        "Sex=0": X.Sex == 0, "Sex=1": X.Sex == 1,
    }
    for group, mask in specs.items():
        idx = np.flatnonzero(mask.to_numpy())
        if len(idx) < 100:
            continue
        for name, p in probs.items():
            score = model_metrics(y[idx], p[idx])
            rows.append({"group": group, "model": name, "n": len(idx),
                         "observed_positive_rate": float(y[idx].mean()),
                         "mean_score": float(p[idx].mean()),
                         "false_negative_rate": 1-score["sensitivity"],
                         "false_positive_rate": 1-score["specificity"],
                         "auc": score["auc"]})
    return pd.DataFrame(rows)


def edge_case_table(X, y, probs):
    rows = []
    for name, p in probs.items():
        pred = (p >= .5).astype(int)
        types = {
            "confident_false_negative": np.flatnonzero((y == 1) & (pred == 0))[
                np.argsort(p[(y == 1) & (pred == 0)])[:5]],
            "confident_false_positive": np.flatnonzero((y == 0) & (pred == 1))[
                np.argsort(-p[(y == 0) & (pred == 1)])[:5]],
            "near_threshold": np.argsort(np.abs(p - .5))[:5],
        }
        for kind, indices in types.items():
            for idx in indices:
                rows.append({"model": name, "case_type": kind, "holdout_index": int(idx),
                             "label": int(y[idx]), "score": float(p[idx]),
                             **X.iloc[idx].to_dict()})
    return pd.DataFrame(rows)


def performance_plots(y, probs, metrics, slices):
    _, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax[0].plot(fpr, tpr, label=f"{name} (AUC {metrics[name]['auc']:.3f})")
        obs, pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax[1].plot(pred, obs, marker="o", label=name)
    ax[0].plot([0, 1], [0, 1], "k--", alpha=.4)
    ax[1].plot([0, 1], [0, 1], "k--", alpha=.4)
    ax[0].set(xlabel="False positive rate", ylabel="Sensitivity", title="Discrimination on shared holdout")
    ax[1].set(xlabel="Average model score", ylabel="Observed positive fraction", title="Calibration within balanced sample")
    ax[0].legend(); ax[1].legend()
    width = .35
    labels = ["True negatives", "False positives", "False negatives", "True positives"]
    for j, (name, result) in enumerate(metrics.items()):
        cm = np.array(result["confusion_matrix"])
        ax[2].bar(np.arange(4) + (j-.5)*width, [cm[0,0], cm[0,1], cm[1,0], cm[1,1]], width, label=name)
    ax[2].set_xticks(np.arange(4), labels, rotation=30, ha="right")
    ax[2].set(ylabel="People", title="Errors at score threshold 0.50")
    ax[2].legend()
    savefig("01_model_validation.png")

    fn = slices.pivot(index="group", columns="model", values="false_negative_rate")
    _, ax = plt.subplots(figsize=(9, 5))
    fn.plot.barh(ax=ax)
    ax.set(xlabel="Missed positive cases / all positive cases in group",
           ylabel="", title="Where positive cases are missed (balanced test sample)")
    ax.legend(title="Model")
    savefig("02_error_slices.png")


def explain_batch(name, model, X_train_raw, X_test_raw, X_train_scaled,
                  X_test_scaled, indices, feature_names, rng):
    """LIME on raw inputs; SHAP on model's own input representation."""
    n_features = len(feature_names)
    if name == "NN":
        predict = lambda x: np.column_stack((1-predict_nn(model, scaler.transform(x)),
                                              predict_nn(model, scaler.transform(x))))
    else:
        predict = lambda x: model.predict_proba(pd.DataFrame(x, columns=feature_names))
    lime_train = X_train_raw[rng.choice(len(X_train_raw), size=3000, replace=False)]
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        lime_train, feature_names=feature_names, class_names=["No", "Yes"],
        mode="classification", random_state=SEED, discretize_continuous=True,
    )
    lime_values = np.zeros((len(indices), n_features))
    lime_fits = []
    for j, idx in enumerate(indices):
        exp = lime_explainer.explain_instance(
            X_test_raw[idx], predict, labels=(1,), num_features=n_features,
            num_samples=1000,
        )
        for feature_id, weight in exp.as_map()[1]:
            lime_values[j, feature_id] = weight
        lime_fits.append(float(exp.score))
    print(f"{name} LIME complete", flush=True)
    if name == "NN":
        background = torch.as_tensor(X_train_scaled[:N_BACKGROUND], dtype=torch.float32)
        deep = shap.GradientExplainer(PositiveProbability(model).eval(), background)
        vals = deep.shap_values(
            torch.as_tensor(X_test_scaled[indices], dtype=torch.float32),
            nsamples=600, rseed=SEED)
        shap_values = np.asarray(vals).reshape(len(indices), n_features)
        with torch.no_grad():
            shap_base = float(PositiveProbability(model)(background).mean())
        probabilities = predict_nn(model, X_test_scaled[indices])
        reconstruction_error = float(np.max(np.abs(
            shap_base + shap_values.sum(axis=1) - probabilities)))
        shap_unit = "probability points (Gradient SHAP approximation)"
    else:
        background = pd.DataFrame(X_train_raw[:N_BACKGROUND], columns=feature_names)
        tree = shap.TreeExplainer(model, data=background,
                                  feature_perturbation="interventional", model_output="probability")
        vals = tree.shap_values(pd.DataFrame(X_test_raw[indices], columns=feature_names))
        shap_values = np.asarray(vals).reshape(len(indices), n_features)
        shap_base = float(np.asarray(tree.expected_value).ravel()[-1])
        reconstruction_error = float(np.max(np.abs(shap_base + shap_values.sum(axis=1) -
                                                   predict(X_test_raw[indices])[:, 1])))
        shap_unit = "sample probability points"
    print(f"{name} SHAP complete", flush=True)
    return {"LIME": lime_values, "SHAP": shap_values}, {
        "lime_local_fit_r2_median": float(np.median(lime_fits)),
        "lime_local_fit_r2_10th_percentile": float(np.quantile(lime_fits, .1)),
        "shap_max_reconstruction_error": reconstruction_error,
        "shap_unit": shap_unit, "shap_base": shap_base,
    }


def explanation_plots(attributions, indices, X_test, features):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for row, name in enumerate(("XGBoost", "NN")):
        for col, method in enumerate(("SHAP", "LIME")):
            values = attributions[name][method]
            imp = pd.Series(np.abs(values).mean(axis=0), index=features).nlargest(10).sort_values()
            imp.plot.barh(ax=axes[row, col], color="#347da4" if method == "SHAP" else "#d3832b")
            unit = "probability points" if method == "SHAP" else "local surrogate weight"
            axes[row, col].set(xlabel=f"Mean absolute contribution ({unit} scale)",
                               title=f"{name} · {method}")
    fig.suptitle("Global summary of local explanations, same 40 holdout people", fontsize=15)
    savefig("03_global_attributions.png")

    # First case is chosen before looking at explanations: the true positive
    # with XGBoost score closest to 0.75.
    j = 0
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for row, name in enumerate(("XGBoost", "NN")):
        for col, method in enumerate(("SHAP", "LIME")):
            vals = attributions[name][method][j]
            order = np.argsort(np.abs(vals))[-8:]
            colors = ["#b94d45" if v > 0 else "#397da9" for v in vals[order]]
            axes[row, col].barh(np.array(features)[order], vals[order], color=colors)
            axes[row, col].axvline(0, color="black", lw=.7)
            axes[row, col].set(title=f"{name} · {method}", xlabel="Pushes model score up (+) or down (−)")
    fig.suptitle(f"One de-identified survey record, holdout index {indices[j]}", fontsize=15)
    savefig("05_local_case.png")

    plain = {"HighBP": "High blood pressure", "HighChol": "High cholesterol",
             "CholCheck": "Cholesterol check", "BMI": "Body mass index",
             "Smoker": "Ever smoked 100 cigarettes", "Stroke": "Stroke history",
             "HeartDiseaseorAttack": "Heart disease or heart attack", "PhysActivity": "Recent physical activity",
             "Fruits": "Daily fruit", "Veggies": "Daily vegetables",
             "HvyAlcoholConsump": "Heavy alcohol use", "AnyHealthcare": "Health care coverage",
             "NoDocbcCost": "Care delayed by cost", "GenHlth": "Self-rated health category",
             "MentHlth": "Days of poor mental health", "PhysHlth": "Days of poor physical health",
             "DiffWalk": "Difficulty walking", "Sex": "Recorded sex code",
             "Age": "Age category", "Education": "Education category", "Income": "Income category"}
    local = attributions["XGBoost"]["SHAP"][0]
    order = np.argsort(np.abs(local))[-8:]
    case = X_test.iloc[indices[0]]
    labels = []
    for k in order:
        feature = features[k]
        val = case[feature]
        val_text = ("yes" if val == 1 else "no") if feature in {
            "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
            "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
            "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk"} else f"{val:g}"
        labels.append(f"{plain.get(feature, feature)}: {val_text}")
    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.barh(labels, local[order], color=["#bd534a" if v > 0 else "#467ba1" for v in local[order]])
    ax.axvline(0, color="black", lw=.8)
    ax.set(xlabel="Contribution to the model's survey-status score",
           title="What moved this person's model score up or down")
    ax.text(0, -.25, "Blue BMI is relative to this model's reference group, not a weight recommendation.\n"
                    "Colors describe the score, not what changing an answer would achieve.",
            transform=ax.transAxes, ha="left", va="top", fontsize=9)
    savefig("06_patient_explanation.png")


def quantus_evaluation(attributions, indices, X_train_raw, X_test_raw,
                       X_train_scaled, X_test_scaled, tree, nn_model):
    """Faithfulness with realistic donor values and explanation concentration.

    Quantus draws feature subsets; the custom perturbation takes replacement
    values for the entire subset from one observed training record per person.
    A second donor draw is a deliberate baseline sensitivity check.
    """
    rows = []
    wrappers = {"XGBoost": ProbabilityWrapper(tree, "XGBoost").eval(),
                "NN": ProbabilityWrapper(nn_model, "NN").eval()}
    for name in ("XGBoost", "NN"):
        train = X_train_raw if name == "XGBoost" else X_train_scaled
        X = (X_test_raw if name == "XGBoost" else X_test_scaled)[indices].astype(np.float32)
        # Quantus 0.6.0's subset-size assertion assumes a channel axis for
        # tabular data. The singleton axis leaves all 21 feature values intact.
        X = X[:, None, :]
        for method in ("SHAP", "LIME"):
            values = attributions[name][method].astype(np.float32)[:, None, :]
            for donor_seed in (111, 222):
                donor_rng = np.random.default_rng(donor_seed)

                def empirical_replacement(arr, indices, _train=train,
                                          _donor_rng=donor_rng, **kwargs):
                    result = arr.copy()
                    donors = _train[_donor_rng.integers(len(_train), size=len(arr))].copy()
                    for _ in range(12):
                        unchanged = np.all(
                            donors[np.arange(len(arr))[:, None], indices] ==
                            arr[np.arange(len(arr))[:, None], indices], axis=1)
                        if not np.any(unchanged):
                            break
                        donors[unchanged] = _train[_donor_rng.integers(len(_train), size=int(unchanged.sum()))]
                    unchanged = np.all(
                        donors[np.arange(len(arr))[:, None], indices] ==
                        arr[np.arange(len(arr))[:, None], indices], axis=1)
                    for row in np.flatnonzero(unchanged):
                        for feature in indices[row]:
                            alternatives = np.flatnonzero(_train[:, feature] != arr[row, feature])
                            if len(alternatives):
                                donors[row] = _train[_donor_rng.choice(alternatives)]
                                break
                    result[np.arange(len(arr))[:, None], indices] = (
                        donors[np.arange(len(arr))[:, None], indices])
                    return result

                np.random.seed(2026)
                metric = quantus.FaithfulnessCorrelation(
                    nr_runs=35, subset_size=3, abs=False, normalise=False,
                    perturb_func=empirical_replacement, return_aggregate=False,
                    disable_warnings=True)
                score = np.asarray(metric(
                    model=wrappers[name], x_batch=X, y_batch=np.ones(len(X), dtype=int),
                    a_batch=values, channel_first=True, softmax=False,
                    batch_size=len(X)), dtype=float)
                for i, s in enumerate(score):
                    rows.append({"holdout_index": int(indices[i]), "model": name,
                                 "method": method, "metric": "faithfulness_correlation",
                                 "replicate": donor_seed, "value": float(s)})
            sparse = quantus.Sparseness(normalise=False, disable_warnings=True)(
                model=wrappers[name], x_batch=X, y_batch=np.ones(len(X), dtype=int),
                a_batch=values, channel_first=True, softmax=False)
            complex_score = quantus.Complexity(normalise=True, disable_warnings=True)(
                model=wrappers[name], x_batch=X, y_batch=np.ones(len(X), dtype=int),
                a_batch=values, channel_first=True, softmax=False)
            for metric_name, scores in (("sparseness", sparse), ("complexity", complex_score)):
                for i, s in enumerate(scores):
                    rows.append({"holdout_index": int(indices[i]), "model": name,
                                 "method": method, "metric": metric_name,
                                 "replicate": 0, "value": float(s)})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "quantus_scores.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    titles = {"faithfulness_correlation": "Does changing credited inputs move the score?\nHigher is better",
              "sparseness": "How concentrated is the explanation?\nHigher is simpler, not necessarily truer",
              "complexity": "Attribution complexity\nLower is simpler, not necessarily truer"}
    for ax, metric_name in zip(axes, titles):
        part = table[(table.metric == metric_name) & (table.replicate != 222)]
        labels = [f"{n}\n{m}" for n in ("XGBoost", "NN") for m in ("SHAP", "LIME")]
        arrays = [part[(part.model == n) & (part.method == m)].value.dropna().to_numpy()
                  for n in ("XGBoost", "NN") for m in ("SHAP", "LIME")]
        ax.boxplot(arrays, tick_labels=labels, showfliers=False)
        ax.set_title(titles[metric_name])
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("Quantus metrics on the same 40 held-out survey records", fontsize=14)
    savefig("04_quantus_scores.png")
    summary = []
    for (name, method, metric, replicate), group in table.groupby(
            ["model", "method", "metric", "replicate"]):
        v = group.value.dropna().to_numpy()
        summary.append({"model": name, "method": method, "metric": metric,
                        "replicate": int(replicate), "n_finite": len(v),
                        "median": float(np.median(v)) if len(v) else None,
                        "mean": float(np.mean(v)) if len(v) else None})
    return summary


def feature_flipping_plots(attributions, indices, X_train_raw, X_test_raw,
                           X_train_scaled, X_test_scaled, y_test, probs, tree, nn_model):
    """Quantus PixelFlipping with observed donor records and a random-order control.

    This tests whether the highest positive attributions are more informative
    than an arbitrary ordering on shared, correctly detected positive records.
    One donor record per person is held fixed throughout each removal curve.
    """
    positive = np.flatnonzero(
        (y_test[indices] == 1) & (probs["XGBoost"][indices] >= .5) &
        (probs["NN"][indices] >= .5))
    # Both models identify these as positive; the label is also positive.
    selected = positive[:16]
    rows = []
    wrappers = {"XGBoost": ProbabilityWrapper(tree, "XGBoost").eval(),
                "NN": ProbabilityWrapper(nn_model, "NN").eval()}
    for name in ("XGBoost", "NN"):
        train = X_train_raw if name == "XGBoost" else X_train_scaled
        X = (X_test_raw if name == "XGBoost" else X_test_scaled)[indices[selected]].astype(np.float32)
        baseline = probs[name][indices[selected]]
        for donor_seed in (301, 302, 303):
            donors = train[np.random.default_rng(donor_seed).integers(len(train), size=len(X))]

            def replace_with_fixed_donor(arr, indices, _donors=donors, **kwargs):
                changed = arr.copy()
                changed[np.arange(len(arr))[:, None], indices] = (
                    _donors[np.arange(len(arr))[:, None], indices])
                return changed

            metric = quantus.PixelFlipping(
                features_in_step=1, abs=False, normalise=False,
                perturb_func=replace_with_fixed_donor,
                disable_warnings=True)
            for method in ("SHAP", "LIME", "Random order"):
                if method == "Random order":
                    order_rng = np.random.default_rng(900 + donor_seed)
                    values = order_rng.random((len(X), X.shape[1]))
                else:
                    values = attributions[name][method][selected]
                curves = np.asarray(metric(
                    model=wrappers[name], x_batch=X[:, None, :],
                    y_batch=np.ones(len(X), dtype=int),
                    a_batch=values.astype(np.float32)[:, None, :],
                    channel_first=True, softmax=False, batch_size=len(X)), dtype=float)
                curves = np.column_stack((baseline, curves))
                for i, curve in enumerate(curves):
                    for k, score in enumerate(curve):
                        rows.append({"holdout_index": int(indices[selected[i]]),
                                     "model": name, "method": method,
                                     "donor_seed": donor_seed, "features_replaced": k,
                                     "positive_score": float(score)})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "feature_flipping.csv", index=False)
    plot_feature_flipping(table, len(selected))
    return {"n_cases": len(selected), "donor_seeds": [301, 302, 303]}


def plot_feature_flipping(table, n_cases):
    """Render stored flipping runs without repeating expensive perturbations."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    colors = {"SHAP": "#247a9b", "LIME": "#d28330", "Random order": "#777777"}
    for ax, name in zip(axes, ("XGBoost", "NN")):
        for method in ("SHAP", "LIME", "Random order"):
            part = table[(table.model == name) & (table.method == method)]
            # Average each person over donor draws; shade variation between people.
            per_person = part.groupby(["holdout_index", "features_replaced"]).positive_score.mean()
            wide = per_person.unstack("features_replaced")
            mean = wide.mean(axis=0)
            sem = wide.sem(axis=0)
            ax.plot(mean.index, mean, label=method, color=colors[method], lw=2)
            ax.fill_between(mean.index.to_numpy(), (mean - 1.96 * sem).to_numpy(),
                            (mean + 1.96 * sem).to_numpy(), color=colors[method], alpha=.12)
        ax.set(xlabel="Survey features selected for donor replacement", title=name,
               xlim=(0, 21), ylim=(0, 1))
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Mean model positive-class score")
    axes[1].legend(frameon=False)
    fig.suptitle(f"Quantus feature flipping on {n_cases} shared positive cases\n"
                 "Observed donor values; matching values may leave a feature unchanged", fontsize=13)
    fig.text(.5, -.02, "Bands are approximate 95% intervals across people, after averaging donor runs.",
             ha="center", va="top", fontsize=9, color="#45566b")
    savefig("07_feature_flipping.png")


def explanation_stability(attributions, indices, X_train_raw, X_test_raw,
                          X_train_scaled, X_test_scaled, y_test, probs, tree, nn_model,
                          feature_names):
    """Quantus MaxSensitivity after a one-unit, in-range survey answer edit."""
    shared = np.flatnonzero((y_test[indices] == 1) &
                            (probs["XGBoost"][indices] >= .5) &
                            (probs["NN"][indices] >= .5))
    selected = shared[:8]
    mutable = [feature_names.index(f) for f in ("BMI", "MentHlth", "PhysHlth")]
    limits = {j: (float(X_train_raw[:, j].min()), float(X_train_raw[:, j].max()))
              for j in mutable}
    rows = []
    for name, actual in (("XGBoost", tree), ("NN", nn_model)):
        X = (X_test_raw if name == "XGBoost" else X_test_scaled)[indices[selected]].astype(np.float32)
        wrapper = ProbabilityWrapper(actual, name).eval()
        if name == "XGBoost":
            shap_explainer = shap.TreeExplainer(
                actual, data=pd.DataFrame(X_train_raw[:N_BACKGROUND], columns=feature_names),
                feature_perturbation="interventional", model_output="probability")
        else:
            shap_explainer = shap.GradientExplainer(
                PositiveProbability(actual).eval(),
                torch.as_tensor(X_train_scaled[:N_BACKGROUND], dtype=torch.float32))
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train_raw[np.random.default_rng(2027).choice(len(X_train_raw), 3000, replace=False)],
            feature_names=feature_names, class_names=["No", "Yes"], mode="classification",
            random_state=SEED, discretize_continuous=True)

        def explain(model, inputs, targets, method, _name=name, _actual=actual,
                    _shap_explainer=shap_explainer, _lime_explainer=lime_explainer):
            flat = np.asarray(inputs).reshape(len(inputs), -1)
            if method == "SHAP":
                if _name == "XGBoost":
                    vals = _shap_explainer.shap_values(pd.DataFrame(flat, columns=feature_names))
                else:
                    vals = _shap_explainer.shap_values(
                        torch.as_tensor(flat, dtype=torch.float32),
                        nsamples=300, rseed=705)
                return np.asarray(vals).reshape(len(flat), 1, -1)
            raw = flat if _name == "XGBoost" else scaler.inverse_transform(flat)
            if _name == "XGBoost":
                predict = lambda v: _actual.predict_proba(pd.DataFrame(v, columns=feature_names))
            else:
                def predict(v):
                    p = predict_nn(_actual, scaler.transform(v).astype(np.float32))
                    return np.column_stack((1-p, p))
            values = np.zeros_like(flat)
            _lime_explainer.random_state = np.random.RandomState(705)
            for i, sample in enumerate(raw):
                exp = _lime_explainer.explain_instance(
                    sample, predict, labels=(1,), num_features=len(feature_names), num_samples=1000)
                for j, weight in exp.as_map()[1]:
                    values[i, j] = weight
            return values[:, None, :]

        for method in ("SHAP", "LIME"):
            perturb_rng = np.random.default_rng(2028)

            def one_unit_edit(arr, indices, _name=name, _rng=perturb_rng, **kwargs):
                changed = arr.copy()
                for i in range(len(changed)):
                    j = int(_rng.choice(mutable))
                    raw_value = changed[i, j] if _name == "XGBoost" else (
                        changed[i, j] * scaler.scale_[j] + scaler.mean_[j])
                    low, high = limits[j]
                    sign = int(_rng.choice([-1, 1]))
                    if raw_value + sign < low or raw_value + sign > high:
                        sign = -sign
                    changed[i, j] += sign if _name == "XGBoost" else sign / scaler.scale_[j]
                return changed

            metric = quantus.MaxSensitivity(
                nr_samples=5, abs=False, normalise=False,
                perturb_func=one_unit_edit, return_nan_when_prediction_changes=True,
                disable_warnings=True)
            scores = np.asarray(metric(
                model=wrapper, x_batch=X[:, None, :],
                y_batch=np.ones(len(X), dtype=int),
                explain_func=explain, explain_func_kwargs={"method": method},
                channel_first=True, softmax=False, batch_size=len(X)), dtype=float)
            for i, score in enumerate(scores):
                rows.append({"holdout_index": int(indices[selected[i]]),
                             "model": name, "method": method,
                             "max_sensitivity": float(score)})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "explanation_stability.csv", index=False)
    _, ax = plt.subplots(figsize=(8, 4.6))
    labels = [f"{n}\n{m}" for n in ("XGBoost", "NN") for m in ("SHAP", "LIME")]
    arrays = [table[(table.model == n) & (table.method == m)].max_sensitivity.dropna().to_numpy()
              for n in ("XGBoost", "NN") for m in ("SHAP", "LIME")]
    ax.boxplot(arrays, tick_labels=labels, showfliers=True)
    ax.set(ylabel="Maximum relative explanation change (lower is steadier)",
           title="Quantus stability after a one-unit survey answer edit")
    ax.text(0, -.21, "BMI or poor-health days changed by 1; five edits per case.\n"
                    "Only cases retaining the same predicted class are scored.",
            transform=ax.transAxes, ha="left", va="top", fontsize=9)
    savefig("08_explanation_stability.png")
    return table.groupby(["model", "method"]).max_sensitivity.agg(
        ["count", "median"]).reset_index().to_dict(orient="records")


def explanation_agreement(attributions, indices):
    """Descriptive cross-model/method top-five agreement on identical cases."""
    labels = [f"{name} {method}" for name in ("XGBoost", "NN")
              for method in ("SHAP", "LIME")]
    top = {f"{name} {method}": np.argsort(
        np.abs(attributions[name][method]), axis=1)[:, -5:]
        for name in ("XGBoost", "NN") for method in ("SHAP", "LIME")}
    rows = []
    for left in labels:
        for right in labels:
            for i, holdout_index in enumerate(indices):
                overlap = len(set(top[left][i]) & set(top[right][i])) / 5
                rows.append({"holdout_index": int(holdout_index), "first": left,
                             "second": right, "top_five_overlap": overlap})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "explanation_agreement.csv", index=False)
    mean = table.groupby(["first", "second"]).top_five_overlap.mean().unstack().loc[labels, labels]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    image = ax.imshow(mean, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(4), [label.replace(" ", "\n") for label in labels])
    ax.set_yticks(range(4), [label.replace(" ", "\n") for label in labels])
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{mean.iloc[i, j]:.2f}", ha="center", va="center",
                    color="white" if mean.iloc[i, j] > .55 else "#203040")
    fig.colorbar(image, ax=ax, label="Mean fraction of top five features shared")
    ax.set_title("Explanation agreement on the same 40 people\n"
                 "Random independent top-five lists would share 0.24 on average")
    savefig("09_explanation_agreement.png")
    return mean.to_dict()


def evaluation_radar():
    """Five transparent diagnostics, with raw values saved beside the plot.

    The first three axes test model behaviour. The last two describe the
    explanation and must not be interpreted as proof of correctness. This is
    deliberately a profile, not an aggregate score or method ranking.
    """
    q = pd.read_csv(OUT / "quantus_scores.csv")
    flip = pd.read_csv(OUT / "feature_flipping.csv")
    stable = pd.read_csv(OUT / "explanation_stability.csv")
    agreement = pd.read_csv(OUT / "explanation_agreement.csv")
    dimensions = ["Faithfulness\ncorrelation", "Flipping\nwin rate",
                  "Stability", "Sparseness", "Cross-model\nagreement"]
    rows = []
    for name in ("XGBoost", "NN"):
        for method in ("SHAP", "LIME"):
            faith = q[(q.model == name) & (q.method == method) &
                      (q.metric == "faithfulness_correlation")]
            # Donor draws are repeated measurements of the same people.
            per_person = faith.groupby("holdout_index").value.mean().dropna()
            correlation = float(per_person.median())
            rows.append((name, method, dimensions[0], correlation,
                         (correlation + 1) / 2, len(per_person),
                         "Median Pearson r; plotted as (r + 1) / 2"))

            curves = flip[(flip.model == name) &
                          (flip.features_replaced.between(1, 5))]
            early = curves.groupby(["holdout_index", "donor_seed", "method"])
            early = early.positive_score.mean().unstack("method")
            wins = (early[method] < early["Random order"]).astype(float)
            win_rate = float(wins.mean())
            rows.append((name, method, dimensions[1], win_rate, win_rate,
                         len(wins),
                         "Fraction of paired person/donor runs where mean score after 1–5 replacements is below random order"))

            sensitivity = stable[(stable.model == name) &
                                 (stable.method == method)].max_sensitivity.dropna()
            median_sensitivity = float(sensitivity.median())
            rows.append((name, method, dimensions[2], median_sensitivity,
                         1 / (1 + median_sensitivity), len(sensitivity),
                         "Median Quantus MaxSensitivity; plotted as 1 / (1 + sensitivity)"))

            sparse = q[(q.model == name) & (q.method == method) &
                       (q.metric == "sparseness")].value.dropna()
            sparsity = float(sparse.median())
            rows.append((name, method, dimensions[3], sparsity, sparsity,
                         len(sparse), "Median Gini sparseness; descriptive only"))

            other = "NN" if name == "XGBoost" else "XGBoost"
            shared = agreement[(agreement["first"] == f"{name} {method}") &
                               (agreement["second"] == f"{other} {method}")]
            overlap = float(shared.top_five_overlap.mean())
            rows.append((name, method, dimensions[4], overlap, overlap,
                         len(shared), "Mean top-five overlap with the other model; descriptive only"))

    table = pd.DataFrame(rows, columns=["model", "method", "dimension",
                                        "raw_value", "plotted_value", "n",
                                        "definition"])
    assert table.plotted_value.between(0, 1).all()
    table.to_csv(OUT / "evaluation_radar.csv", index=False)
    angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False)
    closed = np.r_[angles, angles[0]]
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.5),
                             subplot_kw={"projection": "polar"})
    colors = {"SHAP": "#18738d", "LIME": "#d17a29"}
    for ax, name in zip(axes, ("XGBoost", "NN")):
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles, dimensions)
        ax.tick_params(axis="x", pad=17, labelsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([.25, .5, .75, 1.0])
        ax.set_yticklabels([".25", ".50", ".75", "1.00"], fontsize=8)
        ax.set_rlabel_position(225)
        ax.grid(color="#cbd5df", linewidth=.8)
        ax.spines["polar"].set_color("#bdcbd8")
        for method in ("SHAP", "LIME"):
            values = table[(table.model == name) &
                           (table.method == method)].plotted_value.to_numpy()
            ax.plot(closed, np.r_[values, values[0]], label=method,
                    color=colors[method], linewidth=2.3, marker="o", markersize=5)
            ax.fill(closed, np.r_[values, values[0]], color=colors[method], alpha=.07)
        ax.set_title(name, pad=28, fontsize=14)
    axes[1].legend(loc="lower right", bbox_to_anchor=(1.29, -.13), frameon=False)
    fig.suptitle("Explanation evaluation profile", fontsize=16, y=1.02)
    fig.text(.5, -.05, "Higher plotted values extend outward. The first three axes test model behaviour or stability; "
             "sparseness and agreement are descriptive.\n"
             "Transforms and sample counts are in evaluation_radar.csv. No overall score is implied.",
             ha="center", va="top", fontsize=9, color="#45566b")
    savefig("10_evaluation_radar.png")
    return table.to_dict(orient="records")


def main():
    global scaler
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    frame = pd.read_csv(DATA)
    features = frame.drop(columns="Diabetes_binary")
    feature_names = features.columns.tolist()
    labels = frame.Diabetes_binary.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=.2, random_state=SEED, stratify=labels)
    X_train_raw = X_train.to_numpy(dtype=float)
    X_test_raw = X_test.to_numpy(dtype=float)
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    scaler = StandardScaler().fit(X_train_raw)
    X_train_scaled = scaler.transform(X_train_raw).astype(np.float32)
    X_test_scaled = scaler.transform(X_test_raw).astype(np.float32)
    tree = joblib.load(ROOT / "XGBOOST/artifacts/xgboost_model.joblib")
    nn_model, losses = train_nn(X_train_scaled, y_train)
    probs = {"XGBoost": tree.predict_proba(X_test)[:, 1],
             "NN": predict_nn(nn_model, X_test_scaled)}
    pd.DataFrame({"holdout_index": np.arange(len(y_test)), "label": y_test,
                  "xgboost_score": probs["XGBoost"], "nn_score": probs["NN"]}).to_csv(
        OUT / "holdout_predictions.csv", index=False)
    metrics = {name: model_metrics(y_test, p) for name, p in probs.items()}
    auc_comparison = paired_auc_interval(y_test, probs["XGBoost"], probs["NN"])
    saved_metrics = json.loads((ROOT / "XGBOOST/artifacts/metrics.json").read_text())
    assert abs(metrics["XGBoost"]["auc"] - saved_metrics["roc_auc"]) < 1e-7
    slices = slices_table(X_test, y_test, probs)
    slices.to_csv(OUT / "error_slices.csv", index=False)
    edge_case_table(X_test, y_test, probs).to_csv(OUT / "edge_cases.csv", index=False)
    performance_plots(y_test, probs, metrics, slices)
    if losses:
        _, ax = plt.subplots(figsize=(7, 3.5)); ax.plot(losses)
        ax.set(xlabel="Epoch", ylabel="Mean training loss", title="Neural network training")
        savefig("00_nn_training.png")
    else:
        # An existing checkpoint without its loss history must not leave an
        # old training curve among newly generated outputs.
        for suffix in (".png", ".svg"):
            (OUT / f"00_nn_training{suffix}").unlink(missing_ok=True)

    # Fixed random holdout sample for approximate global summaries, plus one
    # preselected true positive close to score 0.75 for the local case.
    pred_tree = (probs["XGBoost"] >= .5).astype(int)
    tp = np.flatnonzero((y_test == 1) & (pred_tree == 1))
    representative = int(tp[np.argmin(np.abs(probs["XGBoost"][tp] - .75))])
    pool = np.delete(np.arange(len(y_test)), representative)
    indices = np.r_[representative,
                    rng.choice(pool, size=N_EXPLAIN - 1, replace=False)].astype(int)
    attribution = {}
    diagnostics = {}
    for name, model in (("XGBoost", tree), ("NN", nn_model)):
        attribution[name], diagnostics[name] = explain_batch(
            name, model, X_train_raw, X_test_raw, X_train_scaled,
            X_test_scaled, indices, feature_names, rng)
    explanation_plots(attribution, indices, X_test, feature_names)
    quantus_summary = quantus_evaluation(attribution, indices, X_train_raw,
        X_test_raw, X_train_scaled, X_test_scaled, tree, nn_model)
    flipping = feature_flipping_plots(attribution, indices, X_train_raw,
        X_test_raw, X_train_scaled, X_test_scaled, y_test, probs, tree, nn_model)
    stability = explanation_stability(attribution, indices, X_train_raw,
        X_test_raw, X_train_scaled, X_test_scaled, y_test, probs, tree,
        nn_model, feature_names)
    agreement = explanation_agreement(attribution, indices)
    radar = evaluation_radar()
    np.savez_compressed(OUT / "attributions.npz", indices=indices,
                        XGBoost_LIME=attribution["XGBoost"]["LIME"],
                        XGBoost_SHAP=attribution["XGBoost"]["SHAP"],
                        NN_LIME=attribution["NN"]["LIME"],
                        NN_SHAP=attribution["NN"]["SHAP"])
    report = {"seed": SEED, "data_rows": len(frame), "train_rows": len(X_train),
              "test_rows": len(X_test), "explanation_indices": indices.tolist(),
              "model_metrics": metrics, "auc_comparison": auc_comparison,
              "explanation_diagnostics": diagnostics,
              "quantus_summary": quantus_summary,
              "feature_flipping": flipping, "stability_summary": stability,
              "agreement_summary": agreement, "evaluation_profile": radar,
              "case": {"holdout_index": int(indices[0]), "label": int(y_test[indices[0]]),
                       "features": X_test.iloc[indices[0]].to_dict(),
                       "scores": {name: float(p[indices[0]]) for name,p in probs.items()}},
              "versions": {"quantus": quantus.__version__, "torch": torch.__version__,
                           "shap": shap.__version__}}
    (OUT / "results.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(json.dumps({"model_metrics": metrics, "explanation_diagnostics": diagnostics,
                      "quantus_summary": quantus_summary}, indent=2), flush=True)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
