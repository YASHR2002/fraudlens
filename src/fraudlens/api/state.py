"""Thread-safe live card state for the API.

FastAPI runs synchronous endpoints in a thread pool, so concurrent requests could interleave
"compute features" and "record transaction" for the same card. One lock makes each request's
read-then-write atomic (scoring is ~ms, so contention is negligible at this scale). A set of
recorded ``trans_num`` values makes recording idempotent: a client retry is scored again but
never double-counted in the card's history.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from fraudlens.features.online import CardStateStore, Transaction


@dataclass
class ScoredFeatures:
    features: dict[str, Any]
    recorded: bool
    duplicate: bool


@dataclass
class LiveState:
    """The card-state store plus bookkeeping the API reports."""

    store: CardStateStore
    meta: dict[str, Any] = field(default_factory=dict)
    recorded_ids: set[str] = field(default_factory=set)
    processed: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def features(self, txn: Transaction, trans_num: str, record: bool) -> ScoredFeatures:
        """Features for ``txn``; then record it if asked and not already recorded.

        Raises:
            OutOfOrderError: if ``txn`` is older than the card's latest recorded transaction.
        """
        with self._lock:
            feats = self.store.features(txn)
            duplicate = trans_num in self.recorded_ids
            recorded = False
            if record and not duplicate:
                self.store.record(txn)
                self.recorded_ids.add(trans_num)
                self.processed += 1
                recorded = True
            return ScoredFeatures(features=feats, recorded=recorded, duplicate=duplicate)

    def summary(self) -> dict[str, Any]:
        return {
            "cards": len(self.store.cards),
            "snapshot": self.meta,
            "transactions_recorded_since_start": self.processed,
        }
