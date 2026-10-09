"""Request and response schemas for the scoring API (Pydantic v2, with examples for /docs)."""

from __future__ import annotations

import datetime as dt
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from fraudlens.explain.llm_explainer import ExplanationResult

EXAMPLE_TRANSACTION: dict[str, Any] = {
    "trans_num": "16bf2e46c54369a8eab2214649506425",
    "cc_num": 3560725013359375,
    "trans_ts": "2020-06-21T22:06:39",
    "amt": 24.84,
    "category": "health_fitness",
    "dob": "1969-09-15",
    "city_pop": 23,
    "lat": 31.8599,
    "long": -102.7413,
    "merch_lat": 32.575873,
    "merch_long": -102.60429,
    "update_state": False,
}


class TransactionIn(BaseModel):
    """A raw card transaction, as a payment system would send it (no names or address)."""

    model_config = ConfigDict(json_schema_extra={"examples": [EXAMPLE_TRANSACTION]})

    trans_num: str = Field(min_length=1, max_length=64, description="Unique transaction id.")
    cc_num: int = Field(gt=0, description="Card identifier (grouping key, not a model input).")
    trans_ts: dt.datetime = Field(description="Transaction time, same clock as the training data.")
    amt: float = Field(gt=0, le=1_000_000, description="Amount in USD.")
    category: str = Field(min_length=1, max_length=64, description="Merchant category.")
    dob: dt.date = Field(description="Cardholder date of birth (used only for age).")
    city_pop: int = Field(gt=0, description="Population of the cardholder's city.")
    lat: float = Field(ge=-90, le=90, description="Cardholder home latitude.")
    long: float = Field(ge=-180, le=180, description="Cardholder home longitude.")
    merch_lat: float = Field(ge=-90, le=90, description="Merchant latitude.")
    merch_long: float = Field(ge=-180, le=180, description="Merchant longitude.")
    update_state: bool = Field(
        default=True,
        description="Record this transaction in the card's history (live traffic). Set false "
        "for what-if scoring that must not change the state.",
    )


class PredictionOut(BaseModel):
    """Fast path result."""

    trans_num: str
    score: float = Field(description="Fraud score in [0, 1].")
    flagged: bool
    decision: Literal["flagged", "approved"]
    threshold: float
    model_name: str
    model_version: str
    state_updated: bool
    duplicate: bool = Field(description="True if this trans_num was already recorded.")
    latency_ms: float


class BatchIn(BaseModel):
    """Up to 1,000 transactions, scored in time order."""

    transactions: list[TransactionIn] = Field(min_length=1, max_length=1000)


class BatchItemOut(BaseModel):
    trans_num: str
    result: PredictionOut | None = None
    error: str | None = None


class BatchOut(BaseModel):
    results: list[BatchItemOut]
    latency_ms: float


class ShapFactorOut(BaseModel):
    factor: str
    label: str
    shap_value: float = Field(description="Contribution to the score in log-odds.")
    direction: Literal["toward fraud", "away from fraud"]
    fact: str


class ExplainedPredictionOut(PredictionOut):
    """Fast path result plus SHAP factors and an analyst note."""

    base_value_log_odds: float
    shap_factors: list[ShapFactorOut]
    features: dict[str, Any] = Field(description="The 18 feature values the model received.")
    explanation: ExplanationResult


class ModelInfoOut(BaseModel):
    model_name: str
    model_version: str
    alias: str | None
    model_family: str
    source: str
    run_id: str | None
    trained_utc: str | None
    threshold: float
    threshold_source: Literal["env", "model"]
    metrics: dict[str, float]
    global_importance: dict[str, float] | None
    fairness: dict[str, dict[str, float]] | None
    state: dict[str, Any]
    llm: dict[str, Any]
    categories: dict[str, str] = Field(description="Merchant category codes -> readable names.")
