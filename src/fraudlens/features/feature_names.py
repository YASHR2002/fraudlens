"""The model's feature catalogue: technical names, types, and human-readable descriptions.

Single source of truth used by training (which columns to use), SHAP plots, and the LLM
prompt (how to describe a feature to an analyst). The SQL in ``sql/02_features.sql`` and the
online implementation in :mod:`fraudlens.features.online` must produce exactly these columns.

History features use only transactions *strictly earlier* than the current one on the same
card (same-second transactions are not "before" each other); windows include their far
edge, e.g. the 1-hour window is ``[t - 1h, t)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Kind = Literal["numeric", "count", "flag", "categorical"]


@dataclass(frozen=True)
class Feature:
    """One model feature."""

    name: str
    kind: Kind
    label: str  # short human-readable name, for plots
    description: str  # one sentence, for analysts and the LLM prompt
    unit: str = ""


FEATURES: tuple[Feature, ...] = (
    Feature("amt", "numeric", "Amount", "Transaction amount.", "USD"),
    Feature("log_amt", "numeric", "Log amount", "Natural log of (1 + transaction amount)."),
    Feature("category", "categorical", "Merchant category", "Merchant category of the purchase."),
    Feature("hour", "count", "Hour of day", "Hour of the transaction (0-23)."),
    Feature("day_of_week", "count", "Day of week", "ISO day of week (1 = Monday, 7 = Sunday)."),
    Feature("is_night", "flag", "Night-time", "1 if the transaction is between 22:00 and 05:59."),
    Feature("age_at_txn", "count", "Customer age", "Cardholder's age in whole years.", "years"),
    Feature(
        "log_city_pop", "numeric", "Log city population",
        "Natural log of (1 + population of the cardholder's city).",
    ),
    Feature(
        "distance_km", "numeric", "Distance to merchant",
        "Great-circle distance between the cardholder's home and the merchant.", "km",
    ),
    Feature(
        "card_txn_count_1h", "count", "Card transactions, last hour",
        "Number of earlier transactions on this card in the previous hour.",
    ),
    Feature(
        "card_txn_count_24h", "count", "Card transactions, last 24h",
        "Number of earlier transactions on this card in the previous 24 hours.",
    ),
    Feature(
        "card_txn_count_7d", "count", "Card transactions, last 7 days",
        "Number of earlier transactions on this card in the previous 7 days.",
    ),
    Feature(
        "card_amt_sum_24h", "numeric", "Card spend, last 24h",
        "Total amount spent on this card in the previous 24 hours.", "USD",
    ),
    Feature(
        "card_avg_amt_prior", "numeric", "Card average amount",
        "Average amount of all earlier transactions on this card (empty if none).", "USD",
    ),
    Feature(
        "amt_to_card_avg_ratio", "numeric", "Amount vs card average",
        "This amount divided by the card's earlier average amount (empty if no history).", "x",
    ),
    Feature(
        "secs_since_last_txn", "numeric", "Time since last transaction",
        "Seconds since the card's previous transaction (empty if none).", "s",
    ),
    Feature(
        "is_new_category_for_card", "flag", "New category for card",
        "1 if this card has never been used in this merchant category before.",
    ),
    Feature(
        "card_txn_number", "count", "Card history depth",
        "Number of earlier transactions on this card (all time).",
    ),
)  # fmt: skip

FEATURE_NAMES: list[str] = [f.name for f in FEATURES]
CATEGORICAL_FEATURES: list[str] = [f.name for f in FEATURES if f.kind == "categorical"]
NUMERIC_FEATURES: list[str] = [f.name for f in FEATURES if f.kind != "categorical"]
# Features that are empty (NULL / NaN) when a card has no earlier history.
NULLABLE_FEATURES: list[str] = [
    "card_avg_amt_prior",
    "amt_to_card_avg_ratio",
    "secs_since_last_txn",
]

# Columns stored next to the features but never used as model inputs.
# gender is kept only for the fairness audit (Phase 7); cc_num only as a grouping key.
ID_COLUMNS: list[str] = ["trans_num", "cc_num", "trans_ts", "source"]
TARGET = "is_fraud"
AUDIT_COLUMNS: list[str] = ["gender"]
FEATURE_TABLE_COLUMNS: list[str] = [*ID_COLUMNS, TARGET, *AUDIT_COLUMNS, *FEATURE_NAMES]

_BY_NAME = {f.name: f for f in FEATURES}


def get_feature(name: str) -> Feature:
    """Return the catalogue entry for ``name`` (raises ``KeyError`` if unknown)."""
    return _BY_NAME[name]


def label(name: str) -> str:
    """Human-readable short name for a feature, falling back to the technical name."""
    return _BY_NAME[name].label if name in _BY_NAME else name
