"""Tests for the LLM explainer. Gemini is always mocked: no network calls in tests or CI."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from fraudlens.explain.factors import Factor
from fraudlens.explain.llm_explainer import (
    FORBIDDEN,
    LLMExplainer,
    fallback_note,
    validate_note,
)
from fraudlens.explain.prompts import AnalystNote, ExplanationContext, build_prompt

FACTORS = [
    Factor("card_spend_24h", "Card spend, last 24h", 9.0, "toward fraud",
           "card spent $3,037.01 in the previous 24 hours"),
    Factor("amount_vs_card_average", "Amount vs card average", 1.5, "toward fraud",
           "amount $330.52 is 6.0x this card's average of $54.94"),
    Factor("category", "Merchant category", -1.2, "away from fraud",
           "merchant category is home"),
]  # fmt: skip


def ctx(score: float = 0.98, trans_num: str = "t1") -> ExplanationContext:
    return ExplanationContext(
        trans_num=trans_num, score=score, threshold=0.43, factors=FACTORS, amount=330.52,
        category="home", hour=22, distance_km=98.0,
    )  # fmt: skip


class FakeModels:
    """Stands in for client.models: replays a scripted list of replies or exceptions."""

    def __init__(self, replies: list[Any]) -> None:
        self.replies = list(replies)
        self.calls: list[dict] = []

    def generate_content(self, model: str, contents: str, config: Any) -> Any:
        self.calls.append({"model": model, "contents": contents, "config": config})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(parsed=None, text=json.dumps(reply))


def explainer(replies: list[Any], **kwargs: Any) -> tuple[LLMExplainer, FakeModels]:
    models = FakeModels(replies)
    llm = LLMExplainer(api_key=None, model="gemini-test", client=SimpleNamespace(models=models),
                       retry_delay_seconds=0, **kwargs)  # fmt: skip
    return llm, models


GOOD = {
    "summary": "Flagged: the card spent $3,037.01 in 24 hours and this amount is 6.0x its average.",
    "key_reasons": ["heavy 24-hour spending", "amount 6.0x the card's average"],
    "recommended_action": "review",
}


class APIError(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(f"HTTP {code}")
        self.code = code


# --- prompt ------------------------------------------------------------------------------


def test_prompt_contains_facts_but_no_identifiers() -> None:
    prompt = build_prompt(ctx())
    assert "Decision: FLAGGED" in prompt
    assert "6.0x this card's average" in prompt
    assert "(raised the fraud score)" in prompt and "(lowered the fraud score)" in prompt
    assert "background only" in prompt
    assert not FORBIDDEN.search(prompt)


# --- happy path, cache, retries ----------------------------------------------------------


def test_llm_note_is_used_and_cached() -> None:
    llm, models = explainer([GOOD])
    first = llm.explain(ctx())
    assert first.explanation_source == "llm" and first.llm_model == "gemini-test"
    assert first.recommended_action == "review"
    assert llm.explain(ctx()) == first  # served from cache: no second call
    assert len(models.calls) == 1


def test_transient_error_is_retried_once() -> None:
    llm, models = explainer([TimeoutError("timed out"), GOOD])
    assert llm.explain(ctx()).explanation_source == "llm"
    assert len(models.calls) == 2


def test_two_failures_fall_back_and_are_not_cached() -> None:
    llm, models = explainer([APIError(503), APIError(503), GOOD])
    first = llm.explain(ctx())
    assert first.explanation_source == "fallback" and "503" in first.error
    assert llm.explain(ctx()).explanation_source == "llm"  # recovered: fallback wasn't cached
    assert len(models.calls) == 3


def test_permanent_errors_are_not_retried() -> None:
    llm, models = explainer([APIError(400), GOOD])
    assert llm.explain(ctx()).explanation_source == "fallback"
    assert len(models.calls) == 1


def test_no_api_key_means_fallback_without_any_call() -> None:
    llm = LLMExplainer(api_key=None, model="gemini-test")
    result = llm.explain(ctx())
    assert result.explanation_source == "fallback"
    assert "GOOGLE_API_KEY" in result.error


# --- validation of the LLM reply ---------------------------------------------------------


@pytest.mark.parametrize(
    ("reply", "why"),
    [
        ({**GOOD, "recommended_action": "approve"}, "contradicts a flagged decision"),
        (
            {**GOOD, "summary": "The cardholder's age of 48 makes this suspicious overall."},
            "mentions a protected attribute",
        ),
        (
            {**GOOD, "summary": "Card 4000123412341234 shows unusually heavy activity today."},
            "contains a card number",
        ),
        (
            {"summary": "too short", "key_reasons": ["x"], "recommended_action": "review"},
            "summary too short",
        ),
        ({**GOOD, "recommended_action": "escalate"}, "unknown action"),
    ],
)
def test_invalid_replies_are_rejected(reply: dict, why: str) -> None:
    llm, models = explainer([reply, reply])
    result = llm.explain(ctx())
    assert result.explanation_source == "fallback", why
    assert "invalid LLM reply" in result.error
    assert len(models.calls) == 2  # retried once, then fell back


def test_approved_transaction_must_be_approved() -> None:
    note = AnalystNote(**{**GOOD, "recommended_action": "review"})
    assert validate_note(note, ctx(score=0.01))  # problem reported
    assert not validate_note(AnalystNote(**{**GOOD, "recommended_action": "approve"}), ctx(0.01))


def test_average_is_not_mistaken_for_age() -> None:
    note = AnalystNote(
        **{**GOOD, "summary": "The amount is 6x the average and a large percentage."}
    )
    assert validate_note(note, ctx()) == []


# --- deterministic fallback --------------------------------------------------------------


def test_fallback_for_flagged_transaction() -> None:
    summary, reasons, action = fallback_note(ctx(score=0.98))
    assert action == "review"  # the fallback never auto-blocks
    assert "0.98" in summary and "flagged" in summary
    assert "card spent $3,037.01" in summary
    assert "merchant category is home, which lowered the score" in summary
    assert reasons[0].startswith("Card spent")
    assert not FORBIDDEN.search(summary)


def test_fallback_for_approved_transaction() -> None:
    summary, reasons, action = fallback_note(ctx(score=0.01))
    assert action == "approve"
    assert "approved" in summary
    assert reasons == ["Merchant category is home"]
