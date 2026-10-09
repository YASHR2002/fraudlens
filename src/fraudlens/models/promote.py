"""Promotion gate: decide which registered version (if any) becomes the champion.

Rules (``configs/config.yaml`` > ``promotion``), applied to **validation** metrics at each
version's cost-optimal threshold:

1. recall >= ``min_recall``;
2. precision >= ``min_precision``;
3. PR-AUC >= the current champion's PR-AUC (if a champion exists and the rule is enabled).

Among the versions that pass, the one with the **lowest validation business cost** wins (ties:
higher PR-AUC, then the newer version). Versions that fail stay registered with their reasons.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field

from mlflow import MlflowClient

from fraudlens.config import PromotionConfig
from fraudlens.models.registry import VersionInfo, champion, pending_versions

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GateResult:
    """Outcome of the gate for one candidate version."""

    candidate: VersionInfo
    passed: bool
    reasons: tuple[str, ...]  # one line per rule, pass or fail


@dataclass
class PromotionDecision:
    """What the gate decided for a set of candidates."""

    results: list[GateResult]
    winner: VersionInfo | None
    previous_champion: VersionInfo | None
    notes: list[str] = field(default_factory=list)

    @property
    def promoted(self) -> bool:
        return self.winner is not None


def evaluate_gate(
    candidate: VersionInfo, rules: PromotionConfig, champion_pr_auc: float | None
) -> GateResult:
    """Apply every rule to one candidate (pure function, no MLflow)."""
    reasons, passed = [], True

    def check(ok: bool, text: str) -> None:
        nonlocal passed
        passed &= ok
        reasons.append(f"{'PASS' if ok else 'FAIL'}: {text}")

    check(
        candidate.val_recall >= rules.min_recall,
        f"recall {candidate.val_recall:.4f} >= {rules.min_recall:.2f}",
    )
    check(
        candidate.val_precision >= rules.min_precision,
        f"precision {candidate.val_precision:.4f} >= {rules.min_precision:.2f}",
    )
    if rules.require_pr_auc_at_least_champion and champion_pr_auc is not None:
        check(
            candidate.val_pr_auc >= champion_pr_auc,
            f"PR-AUC {candidate.val_pr_auc:.4f} >= champion's {champion_pr_auc:.4f}",
        )
    else:
        reasons.append("SKIP: no current champion to compare PR-AUC against")
    return GateResult(candidate=candidate, passed=passed, reasons=tuple(reasons))


def choose_winner(results: list[GateResult]) -> VersionInfo | None:
    """Lowest validation cost among passing candidates (ties: higher PR-AUC, newer version)."""
    passing = [r.candidate for r in results if r.passed]
    if not passing:
        return None
    return min(passing, key=lambda c: (c.val_cost, -c.val_pr_auc, -int(c.version)))


def decide(
    candidates: list[VersionInfo], rules: PromotionConfig, current: VersionInfo | None
) -> PromotionDecision:
    """Run the gate on every candidate and pick the winner (pure function)."""
    champion_pr_auc = current.val_pr_auc if current else None
    results = [evaluate_gate(c, rules, champion_pr_auc) for c in candidates]
    return PromotionDecision(
        results=results, winner=choose_winner(results), previous_champion=current
    )


def run_promotion(
    client: MlflowClient, model_name: str, alias: str, rules: PromotionConfig
) -> PromotionDecision:
    """Judge all pending versions, move the alias to the winner, and record every decision.

    Each version gets ``promotion_status`` and ``promotion_reason`` tags; each training run gets
    ``promotion_decision`` and ``promotion_reasons`` tags, so the decision is visible from both.
    """
    current = champion(client, model_name, alias)
    candidates = pending_versions(client, model_name)
    decision = decide(candidates, rules, current)
    if not candidates:
        decision.notes.append("No pending versions to evaluate.")
        return decision

    stamp = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    for result in decision.results:
        c = result.candidate
        if decision.winner and c.version == decision.winner.version:
            status = "champion"
            summary = "promoted: passed every rule with the lowest validation cost"
        elif result.passed:
            status = "rejected"
            summary = (
                f"passed the gate but version {decision.winner.version} has lower validation cost "
                f"(${decision.winner.val_cost:,.0f} vs ${c.val_cost:,.0f})"
            )
        else:
            status, summary = "rejected", "failed the gate"
        reason = summary + " | " + "; ".join(result.reasons)
        client.set_model_version_tag(model_name, c.version, "promotion_status", status)
        client.set_model_version_tag(model_name, c.version, "promotion_reason", reason)
        client.set_model_version_tag(model_name, c.version, "promotion_time_utc", stamp)
        if c.run_id:
            client.set_tag(c.run_id, "promotion_decision", status)
            client.set_tag(c.run_id, "promotion_reasons", reason)

    if decision.winner:
        client.set_registered_model_alias(model_name, alias, decision.winner.version)
        if current:
            client.set_model_version_tag(model_name, current.version, "promotion_status", "retired")
            client.set_model_version_tag(
                model_name, current.version, "retired_by_version", decision.winner.version
            )
        logger.info(
            "Alias @%s -> version %s (%s)", alias, decision.winner.version, decision.winner.family
        )
    else:
        decision.notes.append(
            "No candidate passed; the champion is unchanged."
            if current
            else "No candidate passed and there is no champion yet."
        )
    return decision
