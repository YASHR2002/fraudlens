"""Train a model on the train split, evaluate it on validation, and log everything to MLflow.

Each run records: hyperparameters, seed, split dates, cost assumptions, row counts and fraud
rates, the feature list, package versions, validation (and train) metrics at the
cost-optimal threshold, evaluation plots, and the fitted pipeline (skops format).
The test split is never touched here.
"""

from __future__ import annotations

import logging
import platform
import time
from dataclasses import dataclass, field
from importlib.metadata import version
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline

from fraudlens.config import AppConfig
from fraudlens.data.splits import load_features
from fraudlens.features.feature_names import FEATURE_NAMES, TARGET
from fraudlens.models.estimators import (
    SKOPS_TRUSTED_TYPES,
    build_pipeline,
    feature_importance,
    fit_kwargs,
    prepare_features,
)
from fraudlens.models.evaluate import (
    Evaluation,
    evaluate,
    plot_confusion,
    plot_cost_curve,
    plot_feature_importance,
    plot_pr_curve,
)

logger = logging.getLogger(__name__)

TRACKED_PACKAGES = (
    "fraudlens", "numpy", "pandas", "pyarrow", "scikit-learn", "xgboost", "lightgbm", "skops",
    "mlflow", "optuna",
)  # fmt: skip
# What a model needs to run (logged as its pip requirements instead of the whole project).
MODEL_REQUIREMENTS = ("numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "skops")
DISPLAY_NAMES = {"logreg": "Logistic regression", "xgboost": "XGBoost", "lightgbm": "LightGBM"}


@dataclass
class SplitData:
    """Model inputs, labels and amounts for the train and validation splits."""

    X_train: pd.DataFrame
    y_train: np.ndarray
    amt_train: np.ndarray
    X_val: pd.DataFrame
    y_val: np.ndarray
    amt_val: np.ndarray

    @property
    def scale_pos_weight(self) -> float:
        """Legitimate / fraud ratio in train (class weight for the positive class)."""
        return float((self.y_train == 0).sum() / self.y_train.sum())

    def summary(self) -> dict[str, float]:
        """Row counts and fraud rates, logged as run parameters."""
        return {
            "train_rows": len(self.y_train),
            "train_fraud": int(self.y_train.sum()),
            "train_fraud_rate": round(float(self.y_train.mean()), 6),
            "val_rows": len(self.y_val),
            "val_fraud": int(self.y_val.sum()),
            "val_fraud_rate": round(float(self.y_val.mean()), 6),
        }


@dataclass
class TrainResult:
    """Outcome of one training run."""

    name: str
    run_id: str
    model_uri: str
    pipeline: Pipeline
    val: Evaluation
    train_pr_auc: float
    fit_seconds: float
    params: dict[str, Any] = field(default_factory=dict)


def load_split_data(config: AppConfig) -> SplitData:
    """Load the train and validation splits (never test) as model-ready arrays."""
    df = load_features(config, columns=[*FEATURE_NAMES, TARGET], splits=("train", "validation"))
    train, val = df[df["split"] == "train"], df[df["split"] == "validation"]
    for name, part in (("train", train), ("validation", val)):
        if len(part) == 0 or part[TARGET].sum() == 0:
            raise ValueError(
                f"the {name} split has {len(part):,} rows and no fraud; PR-AUC and cost are "
                "undefined. Check split.train_end / split.validation_end in config.yaml."
            )
    return SplitData(
        X_train=prepare_features(train),
        y_train=train[TARGET].to_numpy(dtype=np.int8),
        amt_train=train["amt"].to_numpy(dtype=float),
        X_val=prepare_features(val),
        y_val=val[TARGET].to_numpy(dtype=np.int8),
        amt_val=val["amt"].to_numpy(dtype=float),
    )


def package_versions() -> dict[str, str]:
    """Versions of the packages that determine a model's behaviour."""
    out = {"python": platform.python_version()}
    for pkg in TRACKED_PACKAGES:
        try:
            out[pkg] = version(pkg)
        except Exception:  # noqa: BLE001 - a missing optional package is just not recorded
            out[pkg] = "not installed"
    return out


def configure_mlflow(tracking_uri: str, experiment_name: str) -> None:
    """Point the MLflow client at the tracking server and experiment."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def train_and_log(
    name: str,
    params: dict[str, Any],
    config: AppConfig,
    data: SplitData,
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
) -> TrainResult:
    """Fit ``name`` on train, evaluate on validation, and log the run to MLflow.

    The caller must have called :func:`configure_mlflow`.
    """
    fp_cost = config.costs.false_positive_cost
    pipeline = build_pipeline(name, params, config.seed, data.scale_pos_weight)
    display = DISPLAY_NAMES.get(name, name)
    versions = package_versions()

    with mlflow.start_run(run_name=run_name or f"{name}-baseline") as run:
        mlflow.set_tags({"model_family": name, **(tags or {})})
        mlflow.log_params({f"model.{k}": v for k, v in params.items()})
        mlflow.log_params(
            {
                "model_family": name,
                "seed": config.seed,
                "split.train_end": str(config.split.train_end),
                "split.validation_end": str(config.split.validation_end),
                "cost.false_positive": fp_cost,
                "cost.false_negative": str(config.costs.false_negative_cost),
                "class_weight.scale_pos_weight": round(data.scale_pos_weight, 3),
                "n_features": len(FEATURE_NAMES),
                **data.summary(),
            }
        )
        mlflow.log_dict({"features": FEATURE_NAMES}, "features.json")
        mlflow.log_dict(versions, "environment/package_versions.json")

        logger.info("Fitting %s on %s rows", display, f"{len(data.y_train):,}")
        start = time.perf_counter()
        pipeline.fit(data.X_train, data.y_train, **fit_kwargs(name))
        fit_seconds = time.perf_counter() - start
        logger.info("Fitted %s in %.1fs", display, fit_seconds)

        val_scores = pipeline.predict_proba(data.X_val)[:, 1]
        train_scores = pipeline.predict_proba(data.X_train)[:, 1]
        val_eval = evaluate(data.y_val, val_scores, data.amt_val, fp_cost)
        train_pr_auc = evaluate(data.y_train, train_scores, data.amt_train, fp_cost).pr_auc
        mlflow.log_metrics(
            {**val_eval.metrics("val"), "train_pr_auc": train_pr_auc, "fit_seconds": fit_seconds}
        )

        importance = feature_importance(pipeline)
        mlflow.log_dict(importance.round(6).to_dict(), "feature_importance.json")
        plots = {
            "plots/pr_curve.png": plot_pr_curve(
                data.y_val, val_scores, val_eval, f"{display}: precision-recall (validation)"
            ),
            "plots/confusion_matrix.png": plot_confusion(
                val_eval.cost_point, f"{display}: validation at the cost-optimal threshold"
            ),
            "plots/cost_curve.png": plot_cost_curve(
                data.y_val, val_scores, data.amt_val, fp_cost, val_eval,
                f"{display}: business cost vs alerts (validation)",
            ),
            "plots/feature_importance.png": plot_feature_importance(
                importance, f"{display}: feature importance"
            ),
        }  # fmt: skip
        for path, fig in plots.items():
            mlflow.log_figure(fig, path)

        sample = data.X_val.dropna().head(200)
        info = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=infer_signature(sample, pipeline.predict_proba(sample)),
            input_example=sample.head(3),
            pyfunc_predict_fn="predict_proba",
            skops_trusted_types=SKOPS_TRUSTED_TYPES[name],
            pip_requirements=[f"{p}=={versions[p]}" for p in MODEL_REQUIREMENTS],
        )
    logger.info(
        "%s: val PR-AUC %.4f, cost $%s at threshold %.4g (run %s)",
        display, val_eval.pr_auc, f"{val_eval.cost_point.total_cost:,.0f}",
        val_eval.cost_point.threshold, run.info.run_id,
    )  # fmt: skip
    return TrainResult(
        name=name,
        run_id=run.info.run_id,
        model_uri=info.model_uri,
        pipeline=pipeline,
        val=val_eval,
        train_pr_auc=train_pr_auc,
        fit_seconds=fit_seconds,
        params=params,
    )


def comparison_table(results: list[TrainResult]) -> pd.DataFrame:
    """One row per model: validation metrics at the cost-optimal threshold, plus references."""
    rows = []
    for r in results:
        cp, f1p = r.val.cost_point, r.val.f1_point
        rows.append(
            {
                "model": DISPLAY_NAMES.get(r.name, r.name),
                "PR-AUC": r.val.pr_auc,
                "ROC-AUC": r.val.roc_auc,
                "threshold": cp.threshold,
                "precision": cp.precision,
                "recall": cp.recall,
                "F1": cp.f1,
                "alerts": cp.alerts,
                "cost ($)": cp.total_cost,
                "saving vs flag-nothing": r.val.cost_saving,
                "F1-max threshold": f1p.threshold,
                "cost at F1-max ($)": f1p.total_cost,
                "train PR-AUC": r.train_pr_auc,
                "fit (s)": r.fit_seconds,
            }
        )
    return pd.DataFrame(rows).set_index("model").sort_values("PR-AUC", ascending=False)
