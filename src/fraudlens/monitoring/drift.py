"""Data and prediction drift: each month of test data against a training reference (Evidently).

Reference: a seeded random sample of the **train** split. Current: each calendar month of the
test set (capped per month for speed). Both include the champion's fraud score, so the report
covers feature drift *and* prediction drift, the early warning that the threshold may no longer
be right. Evidently picks the test per column: on samples this size, Wasserstein distance for
numeric columns and Jensen-Shannon distance for categorical ones, with drift at >= 0.1.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset
from sklearn.pipeline import Pipeline

from fraudlens.config import AppConfig
from fraudlens.data.splits import load_features
from fraudlens.features.feature_names import CATEGORICAL_FEATURES, FEATURE_NAMES, NUMERIC_FEATURES
from fraudlens.models.estimators import prepare_features

logger = logging.getLogger(__name__)

SCORE = "fraud_score"
REFERENCE_FILE = "reference.parquet"


@dataclass(frozen=True)
class ColumnDrift:
    column: str
    method: str
    value: float
    threshold: float

    @property
    def drifted(self) -> bool:
        # p-value tests: drift when p < threshold; distances: drift when distance >= threshold.
        if "p_value" in self.method.lower() or "test" in self.method.lower():
            return self.value < self.threshold
        return self.value >= self.threshold


def with_scores(df: pd.DataFrame, pipeline: Pipeline) -> pd.DataFrame:
    """The 18 features plus the champion's score."""
    out = prepare_features(df)
    out[SCORE] = pipeline.predict_proba(out)[:, 1]
    return out


def build_reference(config: AppConfig, pipeline: Pipeline, n: int, path: Path) -> pd.DataFrame:
    """Seeded sample of the train split, scored, saved for reuse."""
    train = load_features(config, columns=FEATURE_NAMES, splits=("train",))
    ref = with_scores(train.sample(min(n, len(train)), random_state=config.seed), pipeline)
    path.parent.mkdir(parents=True, exist_ok=True)
    ref.to_parquet(path, index=False)
    logger.info("Reference sample: %s train rows -> %s", f"{len(ref):,}", path)
    return ref


def _definition() -> DataDefinition:
    return DataDefinition(
        numerical_columns=[*NUMERIC_FEATURES, SCORE], categorical_columns=CATEGORICAL_FEATURES
    )


def column_drift(snapshot_dict: dict) -> list[ColumnDrift]:
    """Per-column drift results from an Evidently snapshot's ``dict()``."""
    out = []
    for m in snapshot_dict["metrics"]:
        cfg = m.get("config", {})
        if cfg.get("type", "").endswith("ValueDrift"):
            out.append(ColumnDrift(cfg["column"], cfg["method"], float(m["value"]),
                                   float(cfg["threshold"])))  # fmt: skip
    return out


def drift_report(
    reference: pd.DataFrame, current: pd.DataFrame, html_path: Path
) -> list[ColumnDrift]:
    """Run the Evidently data drift preset, save its HTML, return per-column results."""
    dd = _definition()
    snapshot = Report([DataDriftPreset()]).run(
        Dataset.from_pandas(current, data_definition=dd),
        Dataset.from_pandas(reference, data_definition=dd),
    )
    html_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(html_path))
    return column_drift(snapshot.dict())


def monthly_drift(
    config: AppConfig,
    pipeline: Pipeline,
    reference: pd.DataFrame,
    out_dir: Path,
    per_month: int = 20_000,
) -> pd.DataFrame:
    """One HTML report per test month; returns a tidy table of every column's drift result."""
    test = load_features(config, columns=[*FEATURE_NAMES, "is_fraud"], splits=("test",))
    test["month"] = test["trans_ts"].dt.to_period("M").astype(str)
    rows = []
    for month, group in test.groupby("month", sort=True):
        sample = group.sample(min(per_month, len(group)), random_state=config.seed)
        current = with_scores(sample, pipeline)
        results = drift_report(reference, current, out_dir / f"drift_{month}.html")
        mean_score = float(current[SCORE].mean())
        for r in results:
            rows.append({
                "month": month, "column": r.column, "method": r.method, "value": r.value,
                "threshold": r.threshold, "drifted": r.drifted, "rows": len(group),
                "fraud_rate": float(group["is_fraud"].mean()), "mean_score": mean_score,
            })  # fmt: skip
        logger.info("%s: %d of %d columns drifted", month, sum(r.drifted for r in results),
                    len(results))  # fmt: skip
    return pd.DataFrame(rows)


def render_summary(table: pd.DataFrame, reference: pd.DataFrame, ref_fraud_rate: float) -> str:
    """Markdown summary: per month, how many and which columns drifted, plus score drift."""
    months = table.groupby("month").agg(
        rows=("rows", "first"), fraud_rate=("fraud_rate", "first"),
        mean_score=("mean_score", "first"), drifted=("drifted", "sum"), columns=("column", "size"),
    )  # fmt: skip
    score = table[table["column"] == SCORE].set_index("month")
    lines = [
        "# Drift report: test months vs training reference",
        "",
        f"Reference: {len(reference):,} random transactions from the train split (fraud rate "
        f"{ref_fraud_rate:.3%}, mean score {reference[SCORE].mean():.4f}). Each month of the test "
        "set is compared with it (up to 20,000 sampled rows per month). Evidently uses "
        "Wasserstein distance (numeric) and Jensen-Shannon distance (categorical); a column has "
        "drifted when its distance is at least 0.1. "
        "HTML reports: `reports/drift/drift_<month>.html`.",
        "",
        "| Month | Transactions | Fraud rate | Mean score | Score drift (distance) "
        "| Drifted columns | Which |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for month, r in months.iterrows():
        drifted = table[(table["month"] == month) & table["drifted"]].sort_values(
            "value", ascending=False)  # fmt: skip
        names = ", ".join(drifted["column"]) or "none"
        s = score.loc[month]
        lines.append(
            f"| {month} | {int(r.rows):,} | {r.fraud_rate:.3%} | {r.mean_score:.4f} | "
            f"{s['value']:.3f}{' (drift)' if s['drifted'] else ''} | "
            f"{int(r.drifted)} of {int(r.columns)} | {names} |"
        )
    top = table.groupby("column")["value"].max().sort_values(ascending=False).head(6)
    lines += ["", "**Largest distances over all months:** "
              + ", ".join(f"{c} ({v:.3f})" for c, v in top.items()) + "."]  # fmt: skip
    return "\n".join(lines) + "\n"


def reference_fraud_rate(config: AppConfig) -> float:
    """Fraud rate of the whole train split (context for the summary)."""
    y = load_features(config, columns=["is_fraud"], splits=("train",))["is_fraud"]
    return float(np.mean(y))
