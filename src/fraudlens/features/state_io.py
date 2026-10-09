"""Build, save and load the per-card state snapshot the scoring API starts from.

The snapshot holds each card's state after its last transaction in fraudTrain (train and
validation, up to 2020-06-21 12:13): exactly what :class:`CardStateStore` would contain after
replaying all of those transactions in time order, computed with vectorised aggregates instead.
The API then updates it with every transaction it scores. In production this state would live in
Redis or a feature store; here it is a JSON file loaded into memory at startup.
"""

from __future__ import annotations

import gzip
import json
import logging
from collections import deque
from pathlib import Path
from typing import Any

import pandas as pd

from fraudlens.features.online import WEEK_S, CardState, CardStateStore

logger = logging.getLogger(__name__)

STATE_FORMAT_VERSION = 1


def _epoch(ts: pd.Series) -> pd.Series:
    """Naive timestamps -> float seconds since 1970-01-01 (same clock as online.to_epoch)."""
    return (ts - pd.Timestamp("1970-01-01")).dt.total_seconds()


def build_state(transactions: pd.DataFrame) -> CardStateStore:
    """State after all ``transactions`` (columns: cc_num, trans_ts, amt, category).

    Equivalent to replaying them in time order through ``CardStateStore.record``.
    """
    df = transactions[["cc_num", "trans_ts", "amt", "category"]].copy()
    df["t"] = _epoch(df["trans_ts"])
    df["category"] = df["category"].astype(str)
    df = df.sort_values(["cc_num", "t"], kind="stable")

    per_card = df.groupby("cc_num").agg(n_total=("amt", "size"), sum_total=("amt", "sum"),
                                        last_ts=("t", "max"))  # fmt: skip
    df = df.join(per_card["last_ts"], on="cc_num")
    prev = df[df["t"] < df["last_ts"]].groupby("cc_num")["t"].max()
    recent = df[df["t"] >= df["last_ts"] - WEEK_S]  # what record() keeps after eviction
    first_seen = df.groupby(["cc_num", "category"])["t"].min()

    store = CardStateStore()
    recent_by_card = recent.groupby("cc_num")
    for cc_num, row in per_card.iterrows():
        rec = recent_by_card.get_group(cc_num)
        cats = first_seen.loc[cc_num]
        store.cards[int(cc_num)] = CardState(
            recent=deque(zip(rec["t"].tolist(), rec["amt"].tolist(), strict=True)),
            n_total=int(row["n_total"]),
            sum_total=float(row["sum_total"]),
            last_ts=float(row["last_ts"]),
            prev_distinct_ts=float(prev[cc_num]) if cc_num in prev.index else None,
            category_first_seen={str(k): float(v) for k, v in cats.items()},
        )  # fmt: skip
    return store


def state_to_dict(store: CardStateStore, meta: dict[str, Any]) -> dict[str, Any]:
    """JSON-serialisable form of the store."""
    return {
        "format_version": STATE_FORMAT_VERSION,
        "meta": meta,
        "cards": {
            str(cc): {
                "recent": [[t, a] for t, a in s.recent],
                "n_total": s.n_total,
                "sum_total": s.sum_total,
                "last_ts": s.last_ts,
                "prev_distinct_ts": s.prev_distinct_ts,
                "category_first_seen": s.category_first_seen,
            }
            for cc, s in store.cards.items()
        },
    }


def state_from_dict(payload: dict[str, Any]) -> CardStateStore:
    """Inverse of :func:`state_to_dict`."""
    if payload.get("format_version") != STATE_FORMAT_VERSION:
        raise ValueError(f"unsupported card state format {payload.get('format_version')!r}")
    store = CardStateStore()
    for cc, s in payload["cards"].items():
        store.cards[int(cc)] = CardState(
            recent=deque((float(t), float(a)) for t, a in s["recent"]),
            n_total=int(s["n_total"]),
            sum_total=float(s["sum_total"]),
            last_ts=s["last_ts"],
            prev_distinct_ts=s["prev_distinct_ts"],
            category_first_seen={k: float(v) for k, v in s["category_first_seen"].items()},
        )
    return store


def save_state(store: CardStateStore, path: Path, meta: dict[str, Any]) -> None:
    """Write the snapshot as gzipped JSON (no pickle: safe to load)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(state_to_dict(store, meta), fh)


def load_state(path: Path) -> tuple[CardStateStore, dict[str, Any]]:
    """Read a snapshot written by :func:`save_state`; returns the store and its metadata."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        payload = json.load(fh)
    return state_from_dict(payload), payload.get("meta", {})
