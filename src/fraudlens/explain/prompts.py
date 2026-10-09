"""Prompt and response schema for LLM analyst notes.

The model sees only facts computed by the system: the score, threshold, decision, the top SHAP
factors (with their actual values) and a little context. It never sees names, card numbers,
or protected attributes (gender is not a model input; age factors are filtered out first).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from fraudlens.explain.factors import CATEGORY_NAMES, Factor

Action = Literal["approve", "review", "block"]

SYSTEM_INSTRUCTION = """\
You write short notes for credit card fraud analysts, explaining why a fraud-detection model
scored a transaction the way it did.

Rules:
- Use ONLY the facts provided. Do not invent amounts, merchants, locations, history or causes.
- Never mention age, gender, race, religion, nationality or any other protected characteristic,
  and never mention names or card numbers.
- The score is a model risk score, not a certainty; do not claim the transaction IS fraud.
- Write for an analyst: plain, specific, 3 to 5 sentences in "summary". Refer to factors by
  their facts (for example "the amount is 8.9 times this card's average"), not by technical
  feature names or SHAP values.
- Reasons must come ONLY from the "Top factors" list. The "Transaction context" is background:
  never present it as a reason unless the same fact also appears among the top factors.
- "key_reasons": 2 to 4 short phrases, most important first, each matching a top factor.
- "recommended_action" must agree with the decision: if the decision is FLAGGED, choose
  "review" or "block" ("block" only when the score is at least 0.95 and several strong factors
  point toward fraud); if the decision is APPROVED, choose "approve".
Return JSON only, matching the schema."""


class AnalystNote(BaseModel):
    """The structure the LLM must return (validated before use).

    No ``extra="forbid"``: the SDK would send it as ``additionalProperties``, which the Gemini
    API rejects. Unknown keys in a reply are simply ignored when parsing.
    """

    summary: str = Field(min_length=20, max_length=1200)
    key_reasons: list[str] = Field(min_length=1, max_length=5)
    recommended_action: Action


@dataclass(frozen=True)
class ExplanationContext:
    """Everything needed to write a note for one scored transaction."""

    trans_num: str
    score: float
    threshold: float
    factors: list[Factor]  # already filtered: no protected attributes
    amount: float
    category: str
    hour: int
    distance_km: float
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def flagged(self) -> bool:
        return self.score >= self.threshold


def build_prompt(ctx: ExplanationContext) -> str:
    """The user message: facts only, in a fixed layout."""
    decision = "FLAGGED for fraud review" if ctx.flagged else "APPROVED"
    lines = [
        f"Fraud score: {ctx.score:.4f} (threshold {ctx.threshold:.4f})",
        f"Decision: {decision}",
        "",
        "Transaction context (background only, not reasons):",
        f"- amount: ${ctx.amount:,.2f}",
        f"- merchant category: {CATEGORY_NAMES.get(ctx.category, ctx.category)}",
        f"- time: between {ctx.hour:02d}:00 and {ctx.hour:02d}:59",
        f"- distance from cardholder's home to merchant: {ctx.distance_km:.0f} km",
        "",
        "Top factors behind the score (largest effect first):",
    ]
    for f in ctx.factors:
        effect = "raised" if f.direction == "toward fraud" else "lowered"
        lines.append(f"- {f.fact} ({effect} the fraud score)")
    return "\n".join(lines)
