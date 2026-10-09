"""LLM analyst notes (Google Gemini) with validation, a retry, a cache and a template fallback.

Flow for one transaction: cache hit -> return it. Otherwise call Gemini (timeout from config)
and validate the JSON reply; on any error or invalid reply, retry once; if that also fails,
return a deterministic note built from the SHAP factors. The scoring service therefore never
fails because the LLM is slow, down, rate-limited or misbehaving.
"""

from __future__ import annotations

import logging
import re
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ValidationError

from fraudlens.explain.prompts import (
    SYSTEM_INSTRUCTION,
    Action,
    AnalystNote,
    ExplanationContext,
    build_prompt,
)

if TYPE_CHECKING:
    from fraudlens.config import AppConfig, EnvSettings

logger = logging.getLogger(__name__)

# Words the note must not contain (protected attributes) and anything that looks like a card
# number. "age" is matched as a whole word so "average" and "percentage" are fine.
FORBIDDEN = re.compile(
    r"\b(age|aged|years? old|gender|male|female|sex|race|religion|nationality)\b|\d{12,19}",
    re.IGNORECASE,
)


# HTTP errors a retry cannot fix (bad request, bad key, no permission, unknown model). Timeouts,
# 429 (rate limit) and 5xx (overload) are retried once.
NON_RETRYABLE_CODES = frozenset({400, 401, 403, 404})


class ExplanationResult(BaseModel):
    """What the API returns for an explanation."""

    summary: str
    key_reasons: list[str]
    recommended_action: Action
    explanation_source: Literal["llm", "fallback"]
    llm_model: str | None = None
    latency_ms: float | None = None
    error: str | None = None


def validate_note(note: AnalystNote, ctx: ExplanationContext) -> list[str]:
    """Problems that make a syntactically valid note unusable (empty list = OK)."""
    problems = []
    allowed = {"review", "block"} if ctx.flagged else {"approve"}
    if note.recommended_action not in allowed:
        problems.append(
            f"recommended_action {note.recommended_action!r} contradicts the decision "
            f"({'flagged' if ctx.flagged else 'approved'})"
        )
    text = " ".join([note.summary, *note.key_reasons])
    if match := FORBIDDEN.search(text):
        problems.append(f"mentions a forbidden term: {match.group(0)!r}")
    return problems


def _capitalise(s: str) -> str:
    return s[:1].upper() + s[1:]


def fallback_note(ctx: ExplanationContext) -> tuple[str, list[str], Action]:
    """A deterministic note from the SHAP factors (no LLM). Never recommends "block"."""
    raising = [f for f in ctx.factors if f.direction == "toward fraud"]
    lowering = [f for f in ctx.factors if f.direction == "away from fraud"]
    if ctx.flagged:
        main = raising[:3] or ctx.factors[:3]
        summary = (
            f"The model scored this transaction {ctx.score:.2f}, at or above the review "
            f"threshold of {ctx.threshold:.2f}, so it was flagged. The main factors raising the "
            "score: " + "; ".join(f.fact for f in main) + "."
        )
        if lowering:
            summary += f" On the other side, {lowering[0].fact}, which lowered the score."
        summary += " This is a model risk score, not proof of fraud: an analyst should review it."
        return summary, [_capitalise(f.fact) for f in main], "review"
    main = lowering[:3] or ctx.factors[:3]
    summary = (
        f"The model scored this transaction {ctx.score:.4f}, below the review threshold of "
        f"{ctx.threshold:.2f}, so it was approved. The main factors keeping the score low: "
        + "; ".join(f.fact for f in main)
        + "."
    )
    if raising:
        summary += f" {_capitalise(raising[0].fact)}, which raised the score, but not enough."
    return summary, [_capitalise(f.fact) for f in main], "approve"


class LLMExplainer:
    """Gemini-backed analyst notes with retry, validation, LRU cache and fallback.

    Args:
        api_key: Google AI Studio key; if missing, every note is a fallback.
        model: Gemini model name (``GEMINI_MODEL``).
        timeout_seconds: Per-request timeout.
        client: Optional pre-built client (tests inject a fake one).
        max_attempts: 2 = one retry.
        cache_size: Maximum cached LLM notes (fallbacks are not cached).
    """

    def __init__(
        self,
        api_key: str | None,
        model: str | None,
        timeout_seconds: float = 15.0,
        client: Any | None = None,
        max_attempts: int = 2,
        retry_delay_seconds: float = 1.0,
        cache_size: int = 1000,
        thinking_level: str | None = "low",
    ) -> None:
        self.model = model
        self.thinking_level = thinking_level
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.retry_delay_seconds = retry_delay_seconds
        self._cache: OrderedDict[str, ExplanationResult] = OrderedDict()
        self._cache_size = cache_size
        self._client = client
        self._disabled_reason: str | None = None
        if client is None:
            if not api_key:
                self._disabled_reason = "GOOGLE_API_KEY is not set"
            elif not model:
                self._disabled_reason = "GEMINI_MODEL is not set"
            else:
                from google import genai
                from google.genai import types

                self._client = genai.Client(
                    api_key=api_key,
                    http_options=types.HttpOptions(timeout=int(timeout_seconds * 1000)),
                )

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def _generation_config(self) -> Any:
        from google.genai import types

        return types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=AnalystNote,
            temperature=0.2,
            max_output_tokens=1024,
            thinking_config=(
                types.ThinkingConfig(thinking_level=self.thinking_level)
                if self.thinking_level
                else None
            ),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    def _call(self, ctx: ExplanationContext) -> AnalystNote:
        response = self._client.models.generate_content(
            model=self.model, contents=build_prompt(ctx), config=self._generation_config()
        )
        parsed = getattr(response, "parsed", None)
        note = (
            parsed
            if isinstance(parsed, AnalystNote)
            else AnalystNote.model_validate_json(response.text or "")
        )
        problems = validate_note(note, ctx)
        if problems:
            raise ValueError("; ".join(problems))
        return note

    def _fallback(self, ctx: ExplanationContext, error: str) -> ExplanationResult:
        summary, reasons, action = fallback_note(ctx)
        return ExplanationResult(
            summary=summary,
            key_reasons=reasons,
            recommended_action=action,
            explanation_source="fallback",
            error=error,
        )

    def explain(self, ctx: ExplanationContext) -> ExplanationResult:
        """An analyst note for ``ctx``: from the LLM if possible, otherwise the fallback."""
        if ctx.trans_num in self._cache:
            self._cache.move_to_end(ctx.trans_num)
            return self._cache[ctx.trans_num]
        if not self.enabled:
            return self._fallback(ctx, self._disabled_reason or "LLM disabled")

        last_error = ""
        for attempt in range(1, self.max_attempts + 1):
            start = time.perf_counter()
            try:
                note = self._call(ctx)
            except (ValidationError, ValueError) as exc:
                last_error = f"invalid LLM reply: {exc}"
            except Exception as exc:  # noqa: BLE001 - timeouts, HTTP and API errors alike
                last_error = f"{type(exc).__name__}: {exc}"
                if getattr(exc, "code", None) in NON_RETRYABLE_CODES:
                    logger.warning("LLM call failed permanently: %s", last_error[:300])
                    break
            else:
                result = ExplanationResult(
                    **note.model_dump(),
                    explanation_source="llm",
                    llm_model=self.model,
                    latency_ms=round((time.perf_counter() - start) * 1000, 1),
                )
                self._cache[ctx.trans_num] = result
                if len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
                return result
            logger.warning("LLM attempt %d/%d failed: %s", attempt, self.max_attempts,
                           last_error[:300])  # fmt: skip
            if attempt < self.max_attempts:
                time.sleep(self.retry_delay_seconds)
        return self._fallback(ctx, last_error[:500])


def llm_from_settings(config: AppConfig, env: EnvSettings) -> LLMExplainer:
    """An LLMExplainer configured from config.yaml and .env (fallback-only without a key)."""
    key = env.google_api_key.get_secret_value() if env.google_api_key else None
    return LLMExplainer(
        api_key=key,
        model=env.gemini_model,
        timeout_seconds=config.llm.timeout_seconds,
        max_attempts=config.llm.max_attempts,
        thinking_level=config.llm.thinking_level,
    )
