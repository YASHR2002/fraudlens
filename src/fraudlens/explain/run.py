"""Phase 7 workflows: explain one transaction, global SHAP plots, fairness audit."""

from __future__ import annotations

import datetime as dt
import logging
import re
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from mlflow import MlflowClient
from sklearn.pipeline import Pipeline

from fraudlens.config import AppConfig, EnvSettings
from fraudlens.data.splits import features_path, load_features
from fraudlens.explain.factors import top_factors
from fraudlens.explain.llm_explainer import ExplanationResult, llm_from_settings
from fraudlens.explain.prompts import ExplanationContext
from fraudlens.explain.shap_explainer import (
    LocalExplanation,
    ShapExplainer,
    global_plots,
    mean_abs_shap,
)
from fraudlens.fairness.audit import audit, render_tables
from fraudlens.features.feature_names import FEATURE_NAMES, TARGET
from fraudlens.models.estimators import prepare_features
from fraudlens.models.registry import VersionInfo, champion
from fraudlens.models.train import DISPLAY_NAMES

logger = logging.getLogger(__name__)


def load_champion(config: AppConfig) -> tuple[Pipeline, VersionInfo]:
    """The @champion pipeline and its registry facts (threshold, metrics)."""
    t = config.training
    version = champion(MlflowClient(), t.registered_model_name, t.champion_alias)
    if version is None:
        raise RuntimeError("No champion yet: run `fraudlens promote` first.")
    uri = f"models:/{t.registered_model_name}@{t.champion_alias}"
    return mlflow.sklearn.load_model(uri), version


def context_for(
    trans_num: str, row: dict[str, Any], exp: LocalExplanation, threshold: float, top_n: int
) -> ExplanationContext:
    """Facts for the LLM prompt: protected attributes are filtered out here."""
    return ExplanationContext(
        trans_num=trans_num,
        score=exp.score,
        threshold=threshold,
        factors=top_factors(exp.factors, top_n, exclude_protected=True),
        amount=float(row["amt"]),
        category=str(row["category"]),
        hour=int(row["hour"]),
        distance_km=float(row["distance_km"]),
    )


def find_transaction(config: AppConfig, trans_num: str) -> pd.DataFrame:
    """One transaction's features (any split) from features.parquet."""
    table = ds.dataset(features_path(config)).to_table(
        columns=["trans_num", "source", TARGET, *FEATURE_NAMES],
        filter=ds.field("trans_num") == trans_num,
    )
    if table.num_rows == 0:
        raise KeyError(f"transaction {trans_num!r} not found in {features_path(config).name}")
    return table.to_pandas()


def explain_transaction(
    config: AppConfig, env: EnvSettings, trans_num: str
) -> tuple[pd.Series, LocalExplanation, VersionInfo, ExplanationResult]:
    """SHAP factors and an analyst note for one transaction, scored by the champion."""
    pipeline, version = load_champion(config)
    df = find_transaction(config, trans_num)
    exp = ShapExplainer(pipeline).explain(df[FEATURE_NAMES])
    row = df.iloc[0]
    ctx = context_for(trans_num, prepare_features(df).iloc[0].to_dict(), exp,
                      version.threshold, config.llm.top_factors)  # fmt: skip
    note = llm_from_settings(config, env).explain(ctx)
    return row, exp, version, note


def log_global_shap(config: AppConfig, out_dir: Path) -> tuple[pd.Series, VersionInfo]:
    """Global SHAP plots on a seeded validation sample, logged to the champion's training run."""
    pipeline, version = load_champion(config)
    val = load_features(config, columns=FEATURE_NAMES, splits=("validation",))
    n = min(config.shap.global_sample_size, len(val))
    sample = val.sample(n, random_state=config.seed)[FEATURE_NAMES]
    explainer = ShapExplainer(pipeline)
    family = DISPLAY_NAMES.get(version.family, version.family)
    figures = global_plots(explainer, sample, f"{family} v{version.version}")
    importance = mean_abs_shap(explainer, sample)
    out_dir.mkdir(parents=True, exist_ok=True)
    with mlflow.start_run(run_id=version.run_id):
        for name, fig in figures.items():
            mlflow.log_figure(fig, f"shap/{name}")
            fig.savefig(out_dir / name, dpi=110)
        mlflow.log_dict(importance.round(6).to_dict(), "shap/mean_abs_shap.json")
        mlflow.log_param("shap.global_sample_size", n)
    logger.info("Logged global SHAP plots (%s rows) to run %s", f"{n:,}", version.run_id)
    return importance, version


def _metric_name(text: str) -> str:
    """MLflow-safe metric name: "age band.70+" -> "age_band.70plus"."""
    return re.sub(r"[^A-Za-z0-9_.]", "_", text.replace("+", "plus").replace(" ", "_"))


def run_fairness_audit(config: AppConfig) -> tuple[dict, str, VersionInfo]:
    """Fairness tables on the test set at the production threshold; report + MLflow run."""
    pipeline, version = load_champion(config)
    test = load_features(config, columns=[*FEATURE_NAMES, TARGET, "gender"], splits=("test",))
    scores = pipeline.predict_proba(prepare_features(test))[:, 1]
    flagged = scores >= version.threshold
    y = test[TARGET].to_numpy()
    result = audit(y, flagged, test["gender"].astype(str), test["age_at_txn"])
    family = DISPLAY_NAMES.get(version.family, version.family)
    header = [
        "# Fairness audit",
        "",
        f"Generated {dt.datetime.now():%Y-%m-%d %H:%M} by `fraudlens fairness-audit`. Model: "
        f"{family} ({config.training.registered_model_name} v{version.version}) at its "
        f"production threshold {version.threshold:.4f}, on the test set ({len(y):,} "
        f"transactions, {int(y.sum()):,} fraud). Gender is not a model input; age is.",
        "",
        "- **Recall**: share of a group's fraud that is caught (missed fraud hurts the customer).",
        "- **Precision**: share of a group's alerts that are real fraud.",
        "- **False positive rate**: share of a group's legitimate transactions wrongly flagged "
        "(a blocked card for an innocent customer).",
        "- 95% Wilson intervals in brackets; overlapping intervals mean a gap may be noise.",
        "",
        "## Results",
        "",
    ]
    report = "\n".join(header) + render_tables(result)
    with mlflow.start_run(run_name=f"fairness-audit-v{version.version}"):
        mlflow.set_tags({"stage": "fairness_audit", "model_version": version.version})
        for attribute, table in result["tables"].items():
            for group, r in table.iterrows():
                key = _metric_name(f"{attribute}.{group}")
                mlflow.log_metrics({f"{key}.recall": r.recall, f"{key}.precision": r.precision,
                                    f"{key}.fpr": r.fpr})  # fmt: skip
        mlflow.log_text(report, "fairness.md")
    return result, report, version


def shap_table(exp: LocalExplanation, top_n: int = 8) -> pd.DataFrame:
    """Top factors as a small table for printing."""
    rows = [
        {
            "factor": f.label,
            "SHAP (log-odds)": round(f.shap, 3),
            "effect": f.direction,
            "fact": f.fact,
        }  # fmt: skip
        for f in exp.factors[:top_n]
    ]
    return pd.DataFrame(rows)


def score_sum_check(exp: LocalExplanation) -> float:
    """sigmoid(base + sum of SHAP); equals the score (additivity), shown for transparency."""
    return float(1 / (1 + np.exp(-(exp.base_value + sum(exp.shap_by_feature.values())))))
