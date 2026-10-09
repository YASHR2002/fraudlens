"""Turn feature values and SHAP contributions into plain-English factors for analysts.

Related features are grouped so an explanation does not list the same idea twice: ``amt`` and
``log_amt`` are one "amount" factor, ``hour`` and ``is_night`` one "time of day" factor. SHAP
values are additive, so a group's contribution is the sum of its members'.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from fraudlens.features.feature_names import FEATURE_NAMES

Direction = Literal["toward fraud", "away from fraud"]

# Readable names for the 14 Sparkov merchant categories.
CATEGORY_NAMES = {
    "entertainment": "entertainment",
    "food_dining": "food and dining",
    "gas_transport": "gas and transport",
    "grocery_net": "online grocery",
    "grocery_pos": "in-store grocery",
    "health_fitness": "health and fitness",
    "home": "home",
    "kids_pets": "kids and pets",
    "misc_net": "miscellaneous online",
    "misc_pos": "miscellaneous in-store",
    "personal_care": "personal care",
    "shopping_net": "online shopping",
    "shopping_pos": "in-store shopping",
    "travel": "travel",
}
DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

# Factor groups: name -> member features (every feature appears in exactly one group).
GROUPS: dict[str, tuple[str, ...]] = {
    "amount": ("amt", "log_amt"),
    "time_of_day": ("hour", "is_night"),
    "day_of_week": ("day_of_week",),
    "category": ("category",),
    "amount_vs_card_average": ("amt_to_card_avg_ratio",),
    "card_average_amount": ("card_avg_amt_prior",),
    "card_spend_24h": ("card_amt_sum_24h",),
    "card_txns_1h": ("card_txn_count_1h",),
    "card_txns_24h": ("card_txn_count_24h",),
    "card_txns_7d": ("card_txn_count_7d",),
    "time_since_last": ("secs_since_last_txn",),
    "new_category": ("is_new_category_for_card",),
    "card_history_depth": ("card_txn_number",),
    "distance": ("distance_km",),
    "city_size": ("log_city_pop",),
    "age": ("age_at_txn",),
}
if sorted(f for g in GROUPS.values() for f in g) != sorted(FEATURE_NAMES):
    raise RuntimeError("factor GROUPS must cover every feature exactly once")

# Protected attributes: shown to analysts in SHAP charts, but never given to the LLM or used
# in generated wording (see docs/decisions.md).
PROTECTED_GROUPS = frozenset({"age"})

LABELS = {
    "amount": "Amount",
    "time_of_day": "Time of day",
    "day_of_week": "Day of week",
    "category": "Merchant category",
    "amount_vs_card_average": "Amount vs card average",
    "card_average_amount": "Card's average amount",
    "card_spend_24h": "Card spend, last 24h",
    "card_txns_1h": "Card transactions, last hour",
    "card_txns_24h": "Card transactions, last 24h",
    "card_txns_7d": "Card transactions, last 7 days",
    "time_since_last": "Time since last transaction",
    "new_category": "New category for card",
    "card_history_depth": "Card history depth",
    "distance": "Distance to merchant",
    "city_size": "City size",
    "age": "Cardholder age",
}


@dataclass(frozen=True)
class Factor:
    """One explanation factor for one transaction."""

    group: str
    label: str
    shap: float  # contribution to the fraud score, in log-odds
    direction: Direction
    fact: str  # plain-English statement with the actual values

    def as_dict(self) -> dict[str, Any]:
        return {
            "factor": self.group,
            "label": self.label,
            "shap_value": round(self.shap, 4),
            "direction": self.direction,
            "fact": self.fact,
        }


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _money(v: float) -> str:
    return f"${v:,.2f}"


def _duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} seconds"
    if seconds < 90 * 60:
        return f"{seconds / 60:.0f} minutes"
    if seconds < 48 * 3600:
        return f"{seconds / 3600:.1f} hours"
    return f"{seconds / 86400:.1f} days"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def describe(group: str, row: dict[str, Any]) -> str:
    """A plain-English statement of the facts behind ``group`` for this transaction."""
    amt = float(row["amt"])
    if group == "amount":
        return f"transaction amount is {_money(amt)}"
    if group == "time_of_day":
        hour = int(row["hour"])
        when = "at night" if int(row["is_night"]) else "during the day"
        return f"made between {hour:02d}:00 and {hour:02d}:59 ({when})"
    if group == "day_of_week":
        return f"made on a {DAYS[int(row['day_of_week']) - 1]}"
    if group == "category":
        cat = str(row["category"])
        return f"merchant category is {CATEGORY_NAMES.get(cat, cat)}"
    if group == "amount_vs_card_average":
        ratio, avg = row["amt_to_card_avg_ratio"], row["card_avg_amt_prior"]
        if _missing(ratio):
            return "first transaction on this card, so there is no spending history to compare"
        return f"amount {_money(amt)} is {float(ratio):.1f}x this card's average of {_money(avg)}"
    if group == "card_average_amount":
        avg = row["card_avg_amt_prior"]
        if _missing(avg):
            return "card has no earlier transactions"
        return f"card's average earlier transaction is {_money(avg)}"
    if group == "card_spend_24h":
        return f"card spent {_money(float(row['card_amt_sum_24h']))} in the previous 24 hours"
    if group in ("card_txns_1h", "card_txns_24h", "card_txns_7d"):
        col = {"card_txns_1h": ("card_txn_count_1h", "hour"),
               "card_txns_24h": ("card_txn_count_24h", "24 hours"),
               "card_txns_7d": ("card_txn_count_7d", "7 days")}[group]  # fmt: skip
        n = int(row[col[0]])
        return f"{_plural(n, 'earlier transaction')} on this card in the previous {col[1]}"
    if group == "time_since_last":
        secs = row["secs_since_last_txn"]
        if _missing(secs):
            return "no previous transaction on this card"
        return f"made {_duration(float(secs))} after the card's previous transaction"
    if group == "new_category":
        if int(row["is_new_category_for_card"]):
            return "first time this card has been used in this merchant category"
        return "card has been used in this merchant category before"
    if group == "card_history_depth":
        return f"card has {_plural(int(row['card_txn_number']), 'earlier transaction')}"
    if group == "distance":
        return f"merchant is {float(row['distance_km']):.0f} km from the cardholder's home"
    if group == "city_size":
        pop = math.expm1(float(row["log_city_pop"]))
        return f"cardholder's city has about {pop:,.0f} residents"
    if group == "age":
        return f"cardholder is {int(row['age_at_txn'])} years old"
    raise KeyError(f"unknown factor group {group!r}")


def build_factors(row: dict[str, Any], shap_by_feature: dict[str, float]) -> list[Factor]:
    """All factors for one transaction, largest absolute contribution first."""
    factors = []
    for group, members in GROUPS.items():
        value = float(sum(shap_by_feature[m] for m in members))
        factors.append(
            Factor(
                group=group,
                label=LABELS[group],
                shap=value,
                direction="toward fraud" if value > 0 else "away from fraud",
                fact=describe(group, row),
            )
        )
    return sorted(factors, key=lambda f: abs(f.shap), reverse=True)


def top_factors(factors: list[Factor], n: int = 5, exclude_protected: bool = False) -> list[Factor]:
    """The ``n`` largest factors, optionally without protected attributes (for the LLM)."""
    keep = [f for f in factors if not (exclude_protected and f.group in PROTECTED_GROUPS)]
    return keep[:n]
