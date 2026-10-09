"""Decision thresholds: minimise business cost (the operating rule), or maximise F1 (reference).

Cost model (from ``configs/config.yaml``):

* a **missed fraud** (false negative) costs its transaction amount;
* a **false alarm** (false positive) costs ``false_positive_cost`` (an analyst's review time);
* correctly flagged fraud and correctly approved transactions cost nothing.

A transaction is flagged when ``score >= threshold``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class OperatingPoint:
    """Confusion counts and cost when flagging ``score >= threshold``."""

    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int
    fraud_amount_caught: float
    fraud_amount_missed: float
    false_positive_cost: float

    @property
    def alerts(self) -> int:
        return self.tp + self.fp

    @property
    def precision(self) -> float:
        return self.tp / self.alerts if self.alerts else 0.0

    @property
    def recall(self) -> float:
        positives = self.tp + self.fn
        return self.tp / positives if positives else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def total_cost(self) -> float:
        return self.fraud_amount_missed + self.fp * self.false_positive_cost

    def as_dict(self) -> dict[str, float]:
        """All fields plus derived metrics, for logging."""
        out = {k: float(v) for k, v in asdict(self).items()}
        out.update(
            alerts=float(self.alerts),
            precision=self.precision,
            recall=self.recall,
            f1=self.f1,
            total_cost=self.total_cost,
        )
        return out


def _validate(y: npt.ArrayLike, scores: npt.ArrayLike, amounts: npt.ArrayLike | None = None):
    y = np.asarray(y, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or y.shape != s.shape:
        raise ValueError("y and scores must be 1-D arrays of the same length")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y must contain only 0 and 1")
    if np.isnan(s).any():
        raise ValueError("scores contain NaN")
    a = None
    if amounts is not None:
        a = np.asarray(amounts, dtype=np.float64)
        if a.shape != y.shape:
            raise ValueError("amounts must have the same length as y")
        if (a < 0).any():
            raise ValueError("amounts must be non-negative")
    return y, s, a


def operating_point(
    y: npt.ArrayLike,
    scores: npt.ArrayLike,
    amounts: npt.ArrayLike,
    threshold: float,
    false_positive_cost: float,
) -> OperatingPoint:
    """Confusion counts and cost at one threshold."""
    y, s, a = _validate(y, scores, amounts)
    flag = s >= threshold
    fraud = y == 1
    return OperatingPoint(
        threshold=float(threshold),
        tp=int((flag & fraud).sum()),
        fp=int((flag & ~fraud).sum()),
        fn=int((~flag & fraud).sum()),
        tn=int((~flag & ~fraud).sum()),
        fraud_amount_caught=float(a[flag & fraud].sum()),
        fraud_amount_missed=float(a[~flag & fraud].sum()),
        false_positive_cost=float(false_positive_cost),
    )


def cost_curve(
    y: npt.ArrayLike, scores: npt.ArrayLike, amounts: npt.ArrayLike, false_positive_cost: float
) -> dict[str, np.ndarray]:
    """Cost and confusion counts at every distinct candidate threshold, in one pass.

    Candidates are each distinct score (flag everything scoring at least that much), plus
    ``inf`` (flag nothing). Tied scores are always flagged together. O(n log n).

    Returns:
        Arrays ``threshold, tp, fp, fn, tn, caught, missed, cost``, thresholds descending.
    """
    y, s, a = _validate(y, scores, amounts)
    order = np.argsort(-s, kind="stable")
    s_sorted, y_sorted, a_sorted = s[order], y[order], a[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    caught = np.cumsum(a_sorted * y_sorted)
    # Keep only the last position of each run of equal scores, so ties are flagged together.
    last_of_group = np.r_[s_sorted[1:] != s_sorted[:-1], True]
    thresholds = np.r_[np.inf, s_sorted[last_of_group]]
    tp = np.r_[0, tp[last_of_group]]
    fp = np.r_[0, fp[last_of_group]]
    caught = np.r_[0.0, caught[last_of_group]]
    total_fraud, total_amount = int(y.sum()), float(a[y == 1].sum())
    missed = total_amount - caught
    return {
        "threshold": thresholds,
        "tp": tp,
        "fp": fp,
        "fn": total_fraud - tp,
        "tn": (len(y) - total_fraud) - fp,
        "caught": caught,
        "missed": missed,
        "cost": missed + fp * false_positive_cost,
    }


def best_cost_threshold(
    y: npt.ArrayLike, scores: npt.ArrayLike, amounts: npt.ArrayLike, false_positive_cost: float
) -> OperatingPoint:
    """The threshold with the lowest total business cost.

    Among equal-cost candidates the highest threshold wins (fewest alerts for the same cost).
    """
    curve = cost_curve(y, scores, amounts, false_positive_cost)
    best = int(np.argmin(curve["cost"]))  # argmin returns the first: the highest threshold
    return operating_point(y, scores, amounts, float(curve["threshold"][best]), false_positive_cost)


def best_f1_threshold(
    y: npt.ArrayLike, scores: npt.ArrayLike, amounts: npt.ArrayLike, false_positive_cost: float
) -> OperatingPoint:
    """The threshold with the highest F1 (reported for comparison only)."""
    curve = cost_curve(y, scores, amounts, false_positive_cost)
    tp, fp, fn = curve["tp"], curve["fp"], curve["fn"]
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(tp > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
    best = int(np.argmax(f1))
    return operating_point(y, scores, amounts, float(curve["threshold"][best]), false_positive_cost)
