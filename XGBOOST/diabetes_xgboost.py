"""Train and evaluate an XGBoost diabetes classifier with Hydra configuration."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import joblib
import numpy as np
import optuna
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from scipy.stats import randint, uniform, loguniform
from sklearn.base import clone
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from xgboost import XGBClassifier


HERE = Path(__file__).resolve().parent


def resolve_project_path(path: str) -> Path:
    """Resolve config paths relative to this script, independent of Hydra's cwd."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (HERE / candidate).resolve()


def load_data(cfg: DictConfig) -> tuple[pd.DataFrame, pd.Series]:
    """Load numeric BRFSS features and a binary target with basic validation."""
    path = resolve_project_path(cfg.data.path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_csv(path)
    if cfg.data.target not in frame:
        raise ValueError(f"Target column {cfg.data.target!r} is missing from {path}")
    frame = frame.dropna(subset=[cfg.data.target])
    if cfg.data.max_rows is not None and int(cfg.data.max_rows) < len(frame):
        fraction = int(cfg.data.max_rows) / len(frame)
        frame, _ = train_test_split(
            frame, train_size=fraction, random_state=cfg.seed,
            stratify=frame[cfg.data.target],
        )
    features = frame.drop(columns=[cfg.data.target])
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes):
        raise TypeError("All XGBoost features must be numeric in this dataset")
    labels = frame[cfg.data.target].astype(int)
    if set(labels.unique()) != {0, 1}:
        raise ValueError("The target must contain both binary classes 0 and 1")
    return features, labels


def split_data(
    features: pd.DataFrame, labels: pd.Series, cfg: DictConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a reproducible, stratified holdout shared with the XAI notebook."""
    return train_test_split(
        features, labels, test_size=cfg.data.test_size,
        random_state=cfg.seed, stratify=labels,
    )


def build_model(cfg: DictConfig) -> XGBClassifier:
    parameters = OmegaConf.to_container(cfg.model, resolve=True)
    return XGBClassifier(**parameters, random_state=cfg.seed, n_jobs=cfg.n_jobs)


def random_search_distributions(space: DictConfig) -> dict:
    """Turn Hydra bounds into scipy distributions for RandomizedSearchCV.

    Integer upper bounds are inclusive, so [200, 400] can sample both endpoints.
    """
    distributions = {}
    for name, bounds in space.items():
        low, high = bounds.low, bounds.high
        if bounds.distribution == "int":
            if int(low) != low or int(high) != high or low > high:
                raise ValueError(f"Invalid integer search bounds for {name}: {low}, {high}")
            distributions[name] = randint(int(low), int(high) + 1)
        elif bounds.distribution == "uniform":
            if low >= high:
                raise ValueError(f"Invalid uniform search bounds for {name}: {low}, {high}")
            distributions[name] = uniform(float(low), float(high - low))
        elif bounds.distribution == "loguniform":
            if low <= 0 or low >= high:
                raise ValueError(f"Invalid loguniform search bounds for {name}: {low}, {high}")
            distributions[name] = loguniform(float(low), float(high))
        else:
            raise ValueError(f"Unknown distribution for {name}: {bounds.distribution!r}")
    return distributions


def optuna_suggestions(trial: optuna.Trial, space: DictConfig) -> dict:
    """Sample the same Hydra search space with Optuna's adaptive sampler."""
    parameters = {}
    for name, bounds in space.items():
        low, high = bounds.low, bounds.high
        if bounds.distribution == "int":
            parameters[name] = trial.suggest_int(name, int(low), int(high))
        elif bounds.distribution == "uniform":
            parameters[name] = trial.suggest_float(name, float(low), float(high))
        elif bounds.distribution == "loguniform":
            parameters[name] = trial.suggest_float(
                name, float(low), float(high), log=True
            )
        else:
            raise ValueError(f"Unknown distribution for {name}: {bounds.distribution!r}")
    return parameters


def tune_hyperparameters(
    model: XGBClassifier, X_train: pd.DataFrame, y_train: pd.Series, cfg: DictConfig
) -> RandomizedSearchCV:
    """Search training folds only; the holdout is never used for selection."""
    folds = StratifiedKFold(
        n_splits=cfg.tuning.cv_folds, shuffle=True, random_state=cfg.seed
    )
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=random_search_distributions(cfg.tuning.search_space),
        n_iter=cfg.tuning.n_iter,
        scoring=cfg.tuning.scoring,
        cv=folds,
        random_state=cfg.seed,
        n_jobs=cfg.tuning.n_jobs,
        refit=False,
        verbose=1,
    )
    search.fit(X_train, y_train)
    return search


def fit_with_early_stopping(
    model: XGBClassifier, X_train: pd.DataFrame, y_train: pd.Series,
    cfg: DictConfig, *, seed: int | None = None,
    max_estimators: int | None = None,
) -> tuple[XGBClassifier, int, float | None]:
    """Select boosting rounds on training-only validation, then refit all rows.

    The returned model has exactly the selected number of trees. The validation
    rows are included again only in the final refit, after round selection.
    """
    if not cfg.early_stopping.enabled:
        model.fit(X_train, y_train)
        return model, int(model.n_estimators), None

    X_fit, X_valid, y_fit, y_valid = train_test_split(
        X_train, y_train,
        test_size=cfg.early_stopping.validation_size,
        random_state=cfg.seed if seed is None else seed,
        stratify=y_train,
    )
    limit = int(cfg.early_stopping.max_estimators)
    if max_estimators is not None:
        limit = min(limit, int(max_estimators))
    candidate = clone(model).set_params(
        n_estimators=limit,
        early_stopping_rounds=int(cfg.early_stopping.rounds),
        eval_metric=cfg.early_stopping.eval_metric,
    )
    candidate.fit(X_fit, y_fit, eval_set=[(X_valid, y_valid)], verbose=False)
    selected_estimators = int(candidate.best_iteration) + 1
    validation_score = float(candidate.best_score)

    final_model = clone(model).set_params(
        n_estimators=selected_estimators, early_stopping_rounds=None,
    )
    final_model.fit(X_train, y_train)
    return final_model, selected_estimators, validation_score


def tune_with_optuna(
    X_train: pd.DataFrame, y_train: pd.Series, cfg: DictConfig
) -> optuna.Study:
    """Optimize mean outer-fold ROC-AUC, using an inner split for early stopping."""
    if cfg.tuning.scoring != "roc_auc":
        raise ValueError("Optuna tuning currently requires tuning.scoring=roc_auc")
    space = cfg.tuning.search_space
    folds = StratifiedKFold(
        n_splits=cfg.tuning.cv_folds, shuffle=True, random_state=cfg.seed,
    )

    def objective(trial: optuna.Trial) -> float:
        parameters = optuna_suggestions(trial, space)
        fold_scores = []
        for fold_number, (fit_idx, score_idx) in enumerate(folds.split(X_train, y_train)):
            X_fit, y_fit = X_train.iloc[fit_idx], y_train.iloc[fit_idx]
            X_score, y_score = X_train.iloc[score_idx], y_train.iloc[score_idx]
            model = build_model(cfg).set_params(**parameters)
            # Early stopping uses only an inner slice of this fold's training rows.
            # The outer fold remains unseen until its ROC-AUC is scored.
            if cfg.early_stopping.enabled:
                X_inner, X_valid, y_inner, y_valid = train_test_split(
                    X_fit, y_fit,
                    test_size=cfg.early_stopping.validation_size,
                    random_state=cfg.seed + fold_number,
                    stratify=y_fit,
                )
                model.set_params(
                    n_estimators=min(
                        int(cfg.early_stopping.max_estimators),
                        int(parameters["n_estimators"]),
                    ),
                    early_stopping_rounds=int(cfg.early_stopping.rounds),
                    eval_metric=cfg.early_stopping.eval_metric,
                )
                model.fit(X_inner, y_inner, eval_set=[(X_valid, y_valid)], verbose=False)
            else:
                model.fit(X_fit, y_fit)
            probabilities = model.predict_proba(X_score)[:, 1]
            fold_scores.append(roc_auc_score(y_score, probabilities))
            trial.report(float(np.mean(fold_scores)), step=fold_number)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(fold_scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=cfg.seed),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=cfg.tuning.optuna.pruning_startup_trials,
            n_warmup_steps=1,
        ),
    )
    study.optimize(
        objective, n_trials=cfg.tuning.optuna.n_trials,
        timeout=cfg.tuning.optuna.timeout_seconds,
        n_jobs=1,
    )
    return study


def evaluate_model(
    model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series
) -> dict:
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "test_class_counts": {
            str(label): int((y_test == label).sum()) for label in (0, 1)
        },
    }


def train(cfg: DictConfig) -> dict:
    features, labels = load_data(cfg)
    X_train, X_test, y_train, y_test = split_data(features, labels, cfg)
    model = build_model(cfg)
    best_parameters = None
    best_cv_score = None
    tuning_method = None
    if cfg.tuning.enabled:
        tuning_method = cfg.tuning.method
        if tuning_method == "random":
            search = tune_hyperparameters(model, X_train, y_train, cfg)
            best_parameters = search.best_params_
            best_cv_score = float(search.best_score_)
        elif tuning_method == "optuna":
            study = tune_with_optuna(X_train, y_train, cfg)
            best_parameters = study.best_params
            best_cv_score = float(study.best_value)
        else:
            raise ValueError(f"Unknown tuning method: {tuning_method!r}")
        model.set_params(**best_parameters)

    tuning_tree_limit = (
        int(best_parameters["n_estimators"])
        if tuning_method is not None else None
    )
    model, selected_estimators, validation_score = fit_with_early_stopping(
        model, X_train, y_train, cfg, max_estimators=tuning_tree_limit,
    )

    metrics = evaluate_model(model, X_test, y_test)
    metrics.update(
        seed=int(cfg.seed),
        train_rows=len(X_train), test_rows=len(X_test),
        feature_names=list(features.columns),
        tuning_method=tuning_method,
        best_parameters=best_parameters,
        best_cv_score=best_cv_score,
        early_stopping_enabled=bool(cfg.early_stopping.enabled),
        early_stopping_max_estimators=(
            min(int(cfg.early_stopping.max_estimators), tuning_tree_limit)
            if tuning_tree_limit is not None else int(cfg.early_stopping.max_estimators)
        ),
        selected_estimators=selected_estimators,
        early_stopping_validation_score=validation_score,
    )
    output_dir = resolve_project_path(cfg.output.dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "xgboost_model.joblib")
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    OmegaConf.save(cfg, output_dir / "run_config.yaml")
    return metrics


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    metrics = train(cfg)
    print(json.dumps(metrics, indent=2))
    print(f"Saved model and metrics to {resolve_project_path(cfg.output.dir)}")


if __name__ == "__main__":
    main()
