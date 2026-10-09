"""Fairness audit: does the champion's performance differ across gender and age groups?

Measured on the test set at the production threshold, after the final evaluation (so it
informs no model decision). For each group: recall (share of fraud caught), precision (share
of alerts that are fraud) and false positive rate (share of legitimate customers wrongly
flagged), each with a 95% Wilson confidence interval, because some groups have only a few
hundred frauds and small gaps can be noise. This module measures; it does not "fix".
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

AGE_BINS = (0, 30, 50, 70, 200)
AGE_LABELS = ("under 30", "30-49", "50-69", "70+")
GENDER_LABELS = {"F": "female", "M": "male"}


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion (well-behaved for small counts and 0/1)."""
    if trials == 0:
        return (float("nan"), float("nan"))
    p = successes / trials
    denom = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def age_band(age: pd.Series) -> pd.Series:
    """Age in whole years -> band label (under 30, 30-49, 50-69, 70+)."""
    return pd.cut(age, bins=AGE_BINS, right=False, labels=AGE_LABELS)


def group_metrics(y: np.ndarray, flagged: np.ndarray, groups: pd.Series) -> pd.DataFrame:
    """Per-group counts and rates with 95% intervals."""
    rows = []
    for name in groups.cat.categories if hasattr(groups, "cat") else sorted(groups.unique()):
        m = (groups == name).to_numpy()
        yy, ff = y[m], flagged[m]
        tp = int((ff & (yy == 1)).sum())
        fp = int((ff & (yy == 0)).sum())
        pos, neg = int((yy == 1).sum()), int((yy == 0).sum())
        alerts = tp + fp
        rec_lo, rec_hi = wilson_interval(tp, pos)
        pre_lo, pre_hi = wilson_interval(tp, alerts)
        fpr_lo, fpr_hi = wilson_interval(fp, neg)
        rows.append(
            {
                "group": str(name),
                "transactions": int(m.sum()),
                "fraud": pos,
                "fraud_rate": pos / m.sum() if m.sum() else float("nan"),
                "alerts": alerts,
                "recall": tp / pos if pos else float("nan"),
                "recall_ci": (rec_lo, rec_hi),
                "precision": tp / alerts if alerts else float("nan"),
                "precision_ci": (pre_lo, pre_hi),
                "fpr": fp / neg if neg else float("nan"),
                "fpr_ci": (fpr_lo, fpr_hi),
            }
        )
    return pd.DataFrame(rows).set_index("group")


@dataclass(frozen=True)
class Gap:
    """Largest difference in one metric across the groups of one attribute."""

    attribute: str
    metric: str
    best: str
    worst: str
    best_value: float
    worst_value: float
    ci_overlap: bool  # True if the two groups' 95% intervals overlap (gap may be noise)

    @property
    def difference(self) -> float:
        return abs(self.best_value - self.worst_value)

    @property
    def ratio(self) -> float:
        lo, hi = sorted([self.best_value, self.worst_value])
        return hi / lo if lo else float("inf")


def largest_gap(table: pd.DataFrame, attribute: str, metric: str, higher_is_better: bool) -> Gap:
    """The pair of groups with the best and worst value of ``metric``."""
    values = table[metric].dropna()
    best = values.idxmax() if higher_is_better else values.idxmin()
    worst = values.idxmin() if higher_is_better else values.idxmax()
    b_lo, b_hi = table.loc[best, f"{metric}_ci"]
    w_lo, w_hi = table.loc[worst, f"{metric}_ci"]
    return Gap(
        attribute=attribute,
        metric=metric,
        best=str(best),
        worst=str(worst),
        best_value=float(values[best]),
        worst_value=float(values[worst]),
        ci_overlap=not (b_lo > w_hi or w_lo > b_hi),
    )


def audit(y: np.ndarray, flagged: np.ndarray, gender: pd.Series, age: pd.Series) -> dict:
    """Tables by gender and age band, plus the largest gap per metric."""
    gender_groups = gender.map(GENDER_LABELS).fillna(gender).astype("category")
    tables = {
        "gender": group_metrics(y, flagged, gender_groups),
        "age band": group_metrics(y, flagged, age_band(age)),
    }
    gaps = [
        largest_gap(table, attribute, metric, higher_is_better)
        for attribute, table in tables.items()
        for metric, higher_is_better in (("recall", True), ("precision", True), ("fpr", False))
    ]
    return {"tables": tables, "gaps": gaps}


def _pct(v: float, ci: tuple[float, float] | None = None, digits: int = 1) -> str:
    if v != v:  # NaN
        return "-"
    text = f"{100 * v:.{digits}f}%"
    if ci:
        text += f" ({100 * ci[0]:.{digits}f}-{100 * ci[1]:.{digits}f})"
    return text


def render_tables(result: dict) -> str:
    """Markdown tables (rates with 95% intervals)."""
    lines = []
    for attribute, table in result["tables"].items():
        lines += [
            f"### By {attribute}",
            "",
            "| Group | Transactions | Fraud | Fraud rate | Alerts | Recall (95% CI) | "
            "Precision (95% CI) | False positive rate (95% CI) |",
            "|---|---:|---:|---:|---:|---|---|---|",
        ]
        for group, r in table.iterrows():
            lines.append(
                f"| {group} | {r.transactions:,} | {r.fraud:,} | {_pct(r.fraud_rate, digits=3)} | "
                f"{r.alerts:,} | {_pct(r.recall, r.recall_ci)} | "
                f"{_pct(r.precision, r.precision_ci)} | {_pct(r.fpr, r.fpr_ci, digits=3)} |"
            )
        lines.append("")
    lines += ["### Largest gaps", "", "| Attribute | Metric | Best group | Worst group | Gap | "
              "Intervals overlap? |", "|---|---|---|---|---|---|"]  # fmt: skip
    names = {"recall": "Recall", "precision": "Precision", "fpr": "False positive rate"}
    for g in result["gaps"]:
        digits = 3 if g.metric == "fpr" else 1
        gap = f"{100 * g.difference:.{digits}f} pts"
        if g.metric == "fpr":
            gap += f" ({g.ratio:.1f}x)"
        lines.append(
            f"| {g.attribute} | {names[g.metric]} | {g.best} ({_pct(g.best_value, digits=digits)}) "
            f"| {g.worst} ({_pct(g.worst_value, digits=digits)}) | {gap} | "
            f"{'yes: may be noise' if g.ci_overlap else 'no: a real difference'} |"
        )
    real = [g for g in result["gaps"] if not g.ci_overlap]
    lines += ["", "### Reading the results", ""]
    for g in real:
        digits = 3 if g.metric == "fpr" else 1
        lines.append(
            f"- **{g.attribute}, {names[g.metric].lower()}:** {g.best} "
            f"{_pct(g.best_value, digits=digits)} vs {g.worst} "
            f"{_pct(g.worst_value, digits=digits)}; the 95% intervals do not overlap, so this "
            "gap is unlikely to be chance."
        )
    if not real:
        lines.append("- No gap is larger than its uncertainty: all intervals overlap.")
    lines.append(
        "- This audit measures and reports; it does not adjust the model. Possible causes and "
        "trade-offs are discussed in `docs/model_card.md`."
    )
    return "\n".join(lines) + "\n"
