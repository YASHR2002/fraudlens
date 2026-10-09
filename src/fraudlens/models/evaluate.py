"""Evaluation metrics and plots for a fraud model's scores."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.ticker as mtick
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from fraudlens.features.feature_names import label
from fraudlens.models.threshold import (
    OperatingPoint,
    best_cost_threshold,
    best_f1_threshold,
    cost_curve,
)
from fraudlens.viz import BLUE, INK2, ORANGE, QUIET, SERIES, apply_style

# Plots are built as matplotlib Figure objects without pyplot: no GUI backend is needed,
# nothing accumulates in pyplot's global state during training, and notebooks still render
# a returned Figure inline.


def _figure(width: float, height: float):
    apply_style()
    fig = Figure(figsize=(width, height))
    return fig, fig.subplots()


@dataclass(frozen=True)
class Evaluation:
    """Metrics for one model on one dataset."""

    n: int
    n_fraud: int
    pr_auc: float
    roc_auc: float
    cost_point: OperatingPoint  # the operating rule: cost-optimal threshold
    f1_point: OperatingPoint  # reference only: F1-optimal threshold
    cost_flag_nothing: float  # baseline: approve everything, every fraud is missed

    @property
    def cost_saving(self) -> float:
        """Share of the flag-nothing cost avoided at the cost-optimal threshold."""
        if self.cost_flag_nothing == 0:
            return float("nan")  # no fraud in the data: nothing to save
        return 1 - self.cost_point.total_cost / self.cost_flag_nothing

    def metrics(self, prefix: str) -> dict[str, float]:
        """Flat metric dict for MLflow, e.g. ``val_pr_auc``."""
        cp, fp = self.cost_point, self.f1_point
        out = {
            "pr_auc": self.pr_auc,
            "roc_auc": self.roc_auc,
            "threshold": cp.threshold,
            "precision": cp.precision,
            "recall": cp.recall,
            "f1": cp.f1,
            "cost": cp.total_cost,
            "alerts": cp.alerts,
            "tp": cp.tp,
            "fp": cp.fp,
            "fn": cp.fn,
            "fraud_amount_caught": cp.fraud_amount_caught,
            "fraud_amount_missed": cp.fraud_amount_missed,
            "cost_flag_nothing": self.cost_flag_nothing,
            "cost_saving": self.cost_saving,
            "f1max_threshold": fp.threshold,
            "f1max_f1": fp.f1,
            "f1max_precision": fp.precision,
            "f1max_recall": fp.recall,
            "f1max_cost": fp.total_cost,
        }
        return {f"{prefix}_{k}": float(v) for k, v in out.items()}


def evaluate(
    y: npt.ArrayLike, scores: npt.ArrayLike, amounts: npt.ArrayLike, false_positive_cost: float
) -> Evaluation:
    """Threshold-free metrics plus both operating points."""
    y = np.asarray(y)
    amounts = np.asarray(amounts, dtype=float)
    return Evaluation(
        n=len(y),
        n_fraud=int(y.sum()),
        pr_auc=float(average_precision_score(y, scores)),
        roc_auc=float(roc_auc_score(y, scores)),
        cost_point=best_cost_threshold(y, scores, amounts, false_positive_cost),
        f1_point=best_f1_threshold(y, scores, amounts, false_positive_cost),
        cost_flag_nothing=float(amounts[y == 1].sum()),
    )


# ------------------------------------------------------------------------------- plots


def plot_pr_curve(y: npt.ArrayLike, scores: npt.ArrayLike, ev: Evaluation, title: str) -> Figure:
    """Precision-recall curve with the cost-optimal and F1-optimal operating points."""
    precision, recall, _ = precision_recall_curve(y, scores)
    fig, ax = _figure(6.4, 4.6)
    ax.plot(recall, precision, color=BLUE, linewidth=2, label=f"PR-AUC {ev.pr_auc:.3f}")
    for point, color, name in [
        (ev.cost_point, ORANGE, "cost-optimal"),
        (ev.f1_point, INK2, "F1-optimal"),
    ]:
        ax.scatter(point.recall, point.precision, s=60, color=color, zorder=3,
                   edgecolor="white", linewidth=1.5,
                   label=f"{name} (t={point.threshold:.3g}): P {point.precision:.2f}, "
                         f"R {point.recall:.2f}")  # fmt: skip
    base = ev.n_fraud / ev.n
    ax.axhline(base, color=QUIET, linestyle="--", linewidth=1)
    ax.text(1.0, base, f"no-skill {base:.2%} ", ha="right", va="bottom", color=INK2, fontsize=8.5)
    ax.set(xlim=(0, 1.01), ylim=(0, 1.02), xlabel="Recall (share of fraud caught)",
           ylabel="Precision (share of alerts that are fraud)")  # fmt: skip
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=8.5)
    return fig


def plot_confusion(point: OperatingPoint, title: str) -> Figure:
    """Confusion matrix at one threshold, with counts and money."""
    fig, ax = _figure(6.0, 3.6)
    ax.set_axis_off()
    cells = [
        (0, 1, "Fraud caught", f"{point.tp:,}", f"${point.fraud_amount_caught:,.0f} saved", BLUE),
        (1, 1, "Fraud missed", f"{point.fn:,}", f"${point.fraud_amount_missed:,.0f} lost", ORANGE),
        (0, 0, "False alarms", f"{point.fp:,}",
         f"${point.fp * point.false_positive_cost:,.0f} review cost", ORANGE),
        (1, 0, "Correctly approved", f"{point.tn:,}", "", QUIET),
    ]  # fmt: skip
    for col, row, name, count, money, color in cells:
        x, yy = 0.03 + col * 0.49, 0.47 - row * 0.45
        ax.add_patch(Rectangle((x, yy), 0.46, 0.4, color=color, alpha=0.12,
                                   transform=ax.transAxes))  # fmt: skip
        ax.text(x + 0.02, yy + 0.31, name, transform=ax.transAxes, fontsize=9.5, color=INK2)
        ax.text(x + 0.02, yy + 0.14, count, transform=ax.transAxes, fontsize=16, weight="bold")
        ax.text(x + 0.02, yy + 0.04, money, transform=ax.transAxes, fontsize=8.5, color=INK2)
    ax.text(0.03, 0.93, "Flagged", transform=ax.transAxes, fontsize=9, color=INK2, weight="bold")
    ax.text(0.52, 0.93, "Approved", transform=ax.transAxes, fontsize=9, color=INK2, weight="bold")
    ax.set_title(f"{title}\nthreshold {point.threshold:.3g}, total cost ${point.total_cost:,.0f}")
    return fig


def plot_cost_curve(
    y: npt.ArrayLike,
    scores: npt.ArrayLike,
    amounts: npt.ArrayLike,
    false_positive_cost: float,
    ev: Evaluation,
    title: str,
) -> Figure:
    """Total business cost against the number of alerts raised."""
    curve = cost_curve(y, scores, amounts, false_positive_cost)
    alerts = curve["tp"] + curve["fp"]
    some = alerts > 0  # "flag nothing" (0 alerts) is drawn as the reference line instead
    fig, ax = _figure(6.4, 4.0)
    ax.plot(alerts[some], curve["cost"][some], color=BLUE, linewidth=2)
    best = ev.cost_point
    ax.scatter(best.alerts, best.total_cost, s=60, color=ORANGE, zorder=3, edgecolor="white",
               linewidth=1.5)  # fmt: skip
    ax.annotate(f"minimum ${best.total_cost:,.0f}\n{best.alerts:,} alerts",
                (best.alerts, best.total_cost), xytext=(12, 18), textcoords="offset points",
                fontsize=8.5, color=INK2)  # fmt: skip
    ax.axhline(ev.cost_flag_nothing, color=QUIET, linestyle="--", linewidth=1)
    ax.text(alerts.max(), ev.cost_flag_nothing, f"flag nothing ${ev.cost_flag_nothing:,.0f} ",
            ha="right", va="bottom", color=INK2, fontsize=8.5)  # fmt: skip
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"${v / 1000:,.0f}k"))
    ax.set(xlabel="Alerts raised (log scale)", ylabel="Total cost (missed fraud + reviews)")
    ax.set_title(title)
    return fig


def plot_feature_importance(importance: pd.Series, title: str, top: int = 18) -> Figure:
    """Horizontal bars of normalised feature importance, with readable names."""
    imp = importance.head(top).iloc[::-1]
    fig, ax = _figure(6.4, 0.28 * len(imp) + 1.2)
    ax.barh([label(n) for n in imp.index], imp.to_numpy(), color=BLUE, height=0.7)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.set_title(title)
    return fig


def plot_pr_comparison(
    curves: dict[str, tuple[npt.ArrayLike, npt.ArrayLike]], title: str
) -> Figure:
    """Precision-recall curves of several models (at most three, for colour safety)."""
    if len(curves) > len(SERIES):
        raise ValueError(f"at most {len(SERIES)} curves per chart")
    fig, ax = _figure(6.4, 4.6)
    for (name, (y, scores)), color in zip(curves.items(), SERIES, strict=False):
        precision, recall, _ = precision_recall_curve(y, scores)
        ax.plot(recall, precision, color=color, linewidth=2,
                label=f"{name} (PR-AUC {average_precision_score(y, scores):.3f})")  # fmt: skip
    ax.set(xlim=(0, 1.01), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision")
    ax.set_title(title)
    ax.legend(loc="lower center")  # the low-recall corner holds weak models' curves
    return fig
