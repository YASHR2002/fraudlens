"""One-time evaluation of the champion on the untouched test set (fraudTest).

Everything that could be tuned was fixed on validation beforehand:

* the model (promotion gate) and its decision threshold (version tag ``threshold``);
* the "flag everything over $X" rule baseline: X minimises validation cost, like the model.

The test set only measures. The best threshold *in hindsight* on test is shown for reference,
to quantify drift, but is not used. The champion version is tagged ``test_evaluated_utc``;
running again requires ``force`` and is recorded as a re-evaluation.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass

import mlflow
import numpy as np
import numpy.typing as npt
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import average_precision_score, roc_auc_score

from fraudlens.config import AppConfig
from fraudlens.data.splits import load_features
from fraudlens.features.feature_names import FEATURE_NAMES, TARGET
from fraudlens.models.estimators import prepare_features
from fraudlens.models.evaluate import (
    evaluate,
    plot_confusion,
    plot_cost_curve,
    plot_pr_curve,
)
from fraudlens.models.registry import VersionInfo, champion
from fraudlens.models.threshold import OperatingPoint, best_cost_threshold, operating_point
from fraudlens.models.train import DISPLAY_NAMES

logger = logging.getLogger(__name__)

TEST_EVALUATED_TAG = "test_evaluated_utc"


class AlreadyEvaluatedError(RuntimeError):
    """The champion has already been evaluated on the test set."""


@dataclass(frozen=True)
class FinalResult:
    """Test-set results for the champion and the two baselines."""

    version: VersionInfo
    n: int
    n_fraud: int
    pr_auc: float
    roc_auc: float
    model: OperatingPoint  # at the validation threshold (the real operating rule)
    hindsight: OperatingPoint  # best threshold chosen on test itself (reference only)
    flag_nothing_cost: float
    amount_rule_x: float  # chosen on validation
    amount_rule: OperatingPoint  # flag amt >= X, applied to test
    monthly: pd.DataFrame


def amount_rule_threshold(
    y_val: npt.ArrayLike, amt_val: npt.ArrayLike, false_positive_cost: float
) -> float:
    """The X for "flag every transaction of at least $X" that minimises validation cost."""
    return best_cost_threshold(y_val, amt_val, amt_val, false_positive_cost).threshold


def monthly_breakdown(
    months: pd.Series,
    y: np.ndarray,
    scores: np.ndarray,
    amt: np.ndarray,
    threshold: float,
    false_positive_cost: float,
) -> pd.DataFrame:
    """Per-month volume, fraud rate and model performance at the fixed threshold."""
    rows = []
    for month in sorted(months.unique()):
        m = (months == month).to_numpy()
        pt = operating_point(y[m], scores[m], amt[m], threshold, false_positive_cost)
        rows.append(
            {
                "month": str(month),
                "transactions": int(m.sum()),
                "fraud": int(y[m].sum()),
                "fraud rate": float(y[m].mean()),
                "recall": pt.recall,
                "precision": pt.precision,
                "alerts": pt.alerts,
                "cost ($)": pt.total_cost,
            }
        )
    return pd.DataFrame(rows).set_index("month")


def compute_final_result(
    version: VersionInfo,
    y: np.ndarray,
    scores: np.ndarray,
    amt: np.ndarray,
    months: pd.Series,
    amount_rule_x: float,
    false_positive_cost: float,
) -> FinalResult:
    """All test metrics (pure function: no MLflow, no file access)."""
    return FinalResult(
        version=version,
        n=len(y),
        n_fraud=int(y.sum()),
        pr_auc=float(average_precision_score(y, scores)),
        roc_auc=float(roc_auc_score(y, scores)),
        model=operating_point(y, scores, amt, version.threshold, false_positive_cost),
        hindsight=best_cost_threshold(y, scores, amt, false_positive_cost),
        flag_nothing_cost=float(amt[y == 1].sum()),
        amount_rule_x=amount_rule_x,
        amount_rule=operating_point(y, amt, amt, amount_rule_x, false_positive_cost),
        monthly=monthly_breakdown(months, y, scores, amt, version.threshold, false_positive_cost),
    )


def _money(v: float) -> str:
    return f"${v:,.0f}"


def render_report(r: FinalResult, model_name: str, fp_cost: float, rerun: bool) -> str:
    """Markdown report for ``reports/final_evaluation.md`` and the README results table."""
    v, m = r.version, r.model
    family = DISPLAY_NAMES.get(v.family, v.family)
    saving = 1 - m.total_cost / r.flag_nothing_cost
    rows = [
        ("Champion model", f"{family} ({model_name} v{v.version})"),
        ("Decision threshold (fixed on validation)", f"{v.threshold:.4f}"),
        ("Test transactions / fraud", f"{r.n:,} / {r.n_fraud:,} ({r.n_fraud / r.n:.3%})"),
        ("PR-AUC", f"{r.pr_auc:.4f} (validation {v.val_pr_auc:.4f})"),
        ("ROC-AUC (reference)", f"{r.roc_auc:.4f}"),
        ("Precision at threshold", f"{m.precision:.4f} (validation {v.val_precision:.4f})"),
        ("Recall at threshold", f"{m.recall:.4f} (validation {v.val_recall:.4f})"),
        ("F1 at threshold", f"{m.f1:.4f}"),
        ("Fraud caught", f"{m.tp:,} of {m.tp + m.fn:,} ({_money(m.fraud_amount_caught)})"),
        ("Fraud missed", f"{m.fn:,} ({_money(m.fraud_amount_missed)})"),
        ("False alarms", f"{m.fp:,} ({_money(m.fp * fp_cost)} review cost)"),
        ("Alerts raised", f"{m.alerts:,}"),
        ("Total cost", f"{_money(m.total_cost)} ({saving:.1%} below flagging nothing)"),
    ]
    baselines = [
        ("Champion model at validation threshold", m),
        (f"Rule: flag every transaction >= ${r.amount_rule_x:,.2f}", r.amount_rule),
    ]
    lines = [
        "# Final evaluation on the test set",
        "",
        f"Generated {dt.datetime.now():%Y-%m-%d %H:%M} by `fraudlens evaluate --final`. "
        "The test set (all of fraudTest, 2020-06-21 to 2020-12-31) was not used for any earlier "
        "decision: the model, its threshold and the rule baseline were all fixed on validation.",
    ]
    if rerun:
        lines += [
            "",
            "> **Note:** this is a forced re-run; the test set had already been evaluated.",
        ]
    lines += ["", "## Headline results", "", "| Metric | Value |", "|---|---|"]
    lines += [f"| {k} | {val} |" for k, val in rows]
    lines += [
        "",
        "## Against simple baselines",
        "",
        "| Strategy | Alerts | Fraud caught | Precision | Recall | Missed fraud | "
        "Review cost | Total cost |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| Flag nothing | 0 | 0 | - | 0.000 | {_money(r.flag_nothing_cost)} | $0 | "
        f"{_money(r.flag_nothing_cost)} |",
    ]
    for name, pt in baselines:
        lines.append(
            f"| {name} | {pt.alerts:,} | {pt.tp:,} | {pt.precision:.3f} | {pt.recall:.3f} | "
            f"{_money(pt.fraud_amount_missed)} | {_money(pt.fp * fp_cost)} | "
            f"{_money(pt.total_cost)} |"
        )
    h = r.hindsight
    lines += [
        "",
        f"The rule's ${r.amount_rule_x:,.2f} cut-off was chosen to minimise cost on validation, "
        "the same way as the model's threshold.",
        "",
        "## Threshold drift (reference only)",
        "",
        f"With hindsight, the cost-minimising threshold on test would have been {h.threshold:.4f} "
        f"(total cost {_money(h.total_cost)}, precision {h.precision:.3f}, recall "
        f"{h.recall:.3f}), versus {_money(m.total_cost)} at the validation threshold "
        f"{v.threshold:.4f}. This is not used; it shows how much a re-tuned threshold could "
        "recover and motivates monitoring the threshold in production.",
        "",
        "## By month (at the fixed threshold)",
        "",
    ]
    monthly = r.monthly.copy()
    monthly["fraud rate"] = monthly["fraud rate"].map("{:.3%}".format)
    for col in ("recall", "precision"):
        monthly[col] = monthly[col].map("{:.3f}".format)
    monthly["cost ($)"] = monthly["cost ($)"].map(_money)
    lines += [monthly.to_markdown(), ""]
    return "\n".join(lines)


def run_final_evaluation(
    config: AppConfig, client: MlflowClient, force: bool = False
) -> tuple[FinalResult, str]:
    """Evaluate the current champion once on test; log to MLflow and tag the version.

    Returns the result and the Markdown report.

    Raises:
        RuntimeError: If there is no champion.
        AlreadyEvaluatedError: If the champion was already evaluated and ``force`` is False.
    """
    t = config.training
    version = champion(client, t.registered_model_name, t.champion_alias)
    if version is None:
        raise RuntimeError("No champion yet: run `fraudlens promote` first.")
    tags = client.get_model_version(t.registered_model_name, version.version).tags
    previous = tags.get(TEST_EVALUATED_TAG)
    if previous and not force:
        raise AlreadyEvaluatedError(
            f"v{version.version} was already evaluated on the test set at {previous}. The test "
            "set should be used once; pass --force only if you understand that re-running it "
            "for decisions would make it a second validation set."
        )
    fp_cost = config.costs.false_positive_cost

    df = load_features(
        config, columns=[*FEATURE_NAMES, TARGET], splits=("validation", "test")
    )  # the only place in the project that loads test rows
    val, test = df[df["split"] == "validation"], df[df["split"] == "test"]
    x = amount_rule_threshold(val[TARGET].to_numpy(), val["amt"].to_numpy(), fp_cost)

    model_uri = f"models:/{t.registered_model_name}@{t.champion_alias}"
    pipeline = mlflow.sklearn.load_model(model_uri)
    y, amt = test[TARGET].to_numpy(), test["amt"].to_numpy(dtype=float)
    scores = pipeline.predict_proba(prepare_features(test))[:, 1]
    months = test["trans_ts"].dt.to_period("M").astype(str)
    result = compute_final_result(version, y, scores, amt, months, x, fp_cost)
    report = render_report(result, t.registered_model_name, fp_cost, rerun=bool(previous))

    family = DISPLAY_NAMES.get(version.family, version.family)
    test_eval = evaluate(y, scores, amt, fp_cost)  # for the plots (curves need all thresholds)
    with mlflow.start_run(run_name=f"final-test-evaluation-v{version.version}"):
        mlflow.set_tags({
            "stage": "final_evaluation",
            "model_version": version.version,
            "model_family": version.family,
            "forced_rerun": str(bool(previous)),
        })  # fmt: skip
        m = result.model
        mlflow.log_metrics({
            "test_pr_auc": result.pr_auc, "test_roc_auc": result.roc_auc,
            "test_precision": m.precision, "test_recall": m.recall, "test_f1": m.f1,
            "test_cost": m.total_cost, "test_alerts": m.alerts, "test_tp": m.tp,
            "test_fp": m.fp, "test_fn": m.fn, "test_cost_flag_nothing": result.flag_nothing_cost,
            "test_cost_amount_rule": result.amount_rule.total_cost,
            "amount_rule_x": result.amount_rule_x,
            "test_hindsight_threshold": result.hindsight.threshold,
            "test_hindsight_cost": result.hindsight.total_cost,
        })  # fmt: skip
        mlflow.log_text(report, "final_evaluation.md")
        mlflow.log_figure(plot_confusion(m, f"{family} on the test set (validation threshold)"),
                          "plots/test_confusion_matrix.png")  # fmt: skip
        mlflow.log_figure(plot_pr_curve(y, scores, test_eval, f"{family}: precision-recall (test)"),
                          "plots/test_pr_curve.png")  # fmt: skip
        mlflow.log_figure(
            plot_cost_curve(y, scores, amt, fp_cost, test_eval,
                            f"{family}: business cost vs alerts (test)"),
            "plots/test_cost_curve.png",
        )  # fmt: skip

    stamp = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    for key, value in {
        TEST_EVALUATED_TAG: stamp,
        "test_pr_auc": repr(result.pr_auc),
        "test_cost": repr(result.model.total_cost),
        "test_precision": repr(result.model.precision),
        "test_recall": repr(result.model.recall),
    }.items():
        client.set_model_version_tag(t.registered_model_name, version.version, key, value)
    logger.info(
        "Test set: PR-AUC %.4f, cost %s (flag nothing %s)",
        result.pr_auc, _money(result.model.total_cost), _money(result.flag_nothing_cost),
    )  # fmt: skip
    return result, report
