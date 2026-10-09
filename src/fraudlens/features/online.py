"""Online (pure-Python) feature computation for one incoming transaction.

This mirrors ``sql/02_features.sql`` exactly; ``tests/test_feature_parity.py`` checks that the
two agree, guarding against training-serving skew. The API keeps one :class:`CardState` per
card in memory (in production this would be Redis or a feature store).

Rules shared with the SQL:

* History features use only transactions on the same card that are strictly earlier than the
  current one. Transactions in the same second are not "before" each other.
* Windows include their far edge: the 1-hour window is ``[t - 1h, t)``.
* A card's transactions must arrive in non-decreasing time order.
"""

from __future__ import annotations

import datetime as dt
import math
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from fraudlens.features.feature_names import FEATURE_NAMES

EARTH_RADIUS_KM = 6371.0
HOUR_S = 3_600.0
DAY_S = 86_400.0
WEEK_S = 7 * DAY_S
_EPOCH = dt.datetime(1970, 1, 1)


class OutOfOrderError(ValueError):
    """A transaction is older than one already recorded for the same card."""


@dataclass(frozen=True)
class Transaction:
    """The raw fields needed to compute features for one transaction."""

    cc_num: int
    trans_ts: dt.datetime  # naive, same clock as the training data
    amt: float
    category: str
    dob: dt.date
    city_pop: int
    lat: float
    long: float
    merch_lat: float
    merch_long: float

    @classmethod
    def from_mapping(cls, row: dict[str, Any]) -> Transaction:
        """Build from a dict (e.g. a DataFrame row or an API payload)."""
        ts = row["trans_ts"]
        dob = row["dob"]
        return cls(
            cc_num=int(row["cc_num"]),
            trans_ts=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
            amt=float(row["amt"]),
            category=str(row["category"]),
            dob=dob.date() if isinstance(dob, dt.datetime) else dob,
            city_pop=int(row["city_pop"]),
            lat=float(row["lat"]),
            long=float(row["long"]),
            merch_lat=float(row["merch_lat"]),
            merch_long=float(row["merch_long"]),
        )


def to_epoch(ts: dt.datetime) -> float:
    """Seconds since 1970-01-01 for a naive timestamp (no timezone conversion)."""
    return (ts - _EPOCH).total_seconds()


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km (same formula and radius as the SQL)."""
    a = (
        math.sin(math.radians(lat2 - lat1) / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def age_in_years(on: dt.date, dob: dt.date) -> int:
    """Whole years between ``dob`` and ``on``, matching PostgreSQL ``AGE()``.

    A 29 February birthday counts as reached on 1 March in non-leap years.
    """
    return on.year - dob.year - ((on.month, on.day) < (dob.month, dob.day))


def static_features(txn: Transaction) -> dict[str, Any]:
    """Features that need only the transaction itself."""
    ts = txn.trans_ts
    return {
        "amt": txn.amt,
        "log_amt": math.log1p(txn.amt),
        "category": txn.category,
        "hour": ts.hour,
        "day_of_week": ts.isoweekday(),
        "is_night": int(ts.hour >= 22 or ts.hour < 6),
        "age_at_txn": age_in_years(ts.date(), txn.dob),
        "log_city_pop": math.log1p(txn.city_pop),
        "distance_km": haversine_km(txn.lat, txn.long, txn.merch_lat, txn.merch_long),
    }


@dataclass
class CardState:
    """Everything needed to compute one card's history features.

    Attributes:
        recent: ``(epoch_seconds, amount)`` of transactions in the last 7 days, oldest first.
        n_total: Number of transactions recorded (all time).
        sum_total: Sum of their amounts.
        last_ts: Latest transaction time recorded.
        prev_distinct_ts: Latest transaction time strictly before ``last_ts``.
        category_first_seen: First time each merchant category was used on the card.
    """

    recent: deque[tuple[float, float]] = field(default_factory=deque)
    n_total: int = 0
    sum_total: float = 0.0
    last_ts: float | None = None
    prev_distinct_ts: float | None = None
    category_first_seen: dict[str, float] = field(default_factory=dict)

    def history_features(self, t: float, amt: float, category: str) -> dict[str, Any]:
        """History features for a transaction at epoch ``t``, before it is recorded."""
        if self.last_ts is not None and t < self.last_ts:
            raise OutOfOrderError(f"transaction at {t} is older than last recorded {self.last_ts}")
        n_1h = n_24h = n_7d = 0
        sum_24h = 0.0
        n_same = 0  # recorded transactions in the same second: not "earlier"
        sum_same = 0.0
        for ts, a in self.recent:
            if ts >= t:
                n_same += 1
                sum_same += a
                continue
            n_7d += ts >= t - WEEK_S
            if ts >= t - DAY_S:
                n_24h += 1
                sum_24h += a
            n_1h += ts >= t - HOUR_S

        n_prior = self.n_total - n_same
        avg_prior = (self.sum_total - sum_same) / n_prior if n_prior else None
        if self.last_ts is None:
            prev = None
        elif self.last_ts < t:
            prev = self.last_ts
        else:  # last recorded is in this same second
            prev = self.prev_distinct_ts
        first_seen = self.category_first_seen.get(category)
        return {
            "card_txn_count_1h": n_1h,
            "card_txn_count_24h": n_24h,
            "card_txn_count_7d": n_7d,
            "card_amt_sum_24h": sum_24h,
            "card_avg_amt_prior": avg_prior,
            "amt_to_card_avg_ratio": amt / avg_prior if avg_prior else None,
            "secs_since_last_txn": t - prev if prev is not None else None,
            "is_new_category_for_card": int(first_seen is None or first_seen >= t),
            "card_txn_number": n_prior,
        }

    def record(self, t: float, amt: float, category: str) -> None:
        """Add a transaction at epoch ``t`` to the state (after it has been scored)."""
        if self.last_ts is not None and t < self.last_ts:
            raise OutOfOrderError(f"transaction at {t} is older than last recorded {self.last_ts}")
        self.recent.append((t, amt))
        while self.recent and self.recent[0][0] < t - WEEK_S:
            self.recent.popleft()  # never needed again: later transactions are later still
        self.n_total += 1
        self.sum_total += amt
        if self.last_ts is None or t > self.last_ts:
            self.prev_distinct_ts = self.last_ts
            self.last_ts = t
        self.category_first_seen.setdefault(category, t)


@dataclass
class CardStateStore:
    """In-memory per-card state for the scoring service."""

    cards: dict[int, CardState] = field(default_factory=dict)

    def features(self, txn: Transaction) -> dict[str, Any]:
        """Compute all model features for ``txn`` without changing the state."""
        state = self.cards.get(txn.cc_num) or CardState()
        history = state.history_features(to_epoch(txn.trans_ts), txn.amt, txn.category)
        row = {**static_features(txn), **history}
        return {name: row[name] for name in FEATURE_NAMES}

    def record(self, txn: Transaction) -> None:
        """Add ``txn`` to its card's history."""
        state = self.cards.setdefault(txn.cc_num, CardState())
        state.record(to_epoch(txn.trans_ts), txn.amt, txn.category)

    def process(self, txn: Transaction) -> dict[str, Any]:
        """Compute features for ``txn``, then record it (the scoring-time sequence)."""
        feats = self.features(txn)
        self.record(txn)
        return feats


def replay(
    transactions: Iterable[Transaction], store: CardStateStore | None = None
) -> Iterator[dict[str, Any]]:
    """Process transactions in the given (time) order, yielding each one's features."""
    store = store or CardStateStore()
    for txn in transactions:
        yield store.process(txn)
