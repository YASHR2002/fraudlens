"""Replay test-set transactions against the running API, in time order, as simulated live traffic.

Transactions are sent one at a time at ``speed`` per second (a modest rate keeps the laptop
responsive), with ``update_state=true``, so the API's card history advances exactly as it would in
production and its features stay exact. Every response is saved with the true label, so the run
reports live recall/precision and its predictions can be analysed for drift.

Starting point: the API state begins at the end of fraudTrain. Replaying from the start of
fraudTest continues that history seamlessly; a transaction older than its card's latest one is
rejected by the API (409) and counted, e.g. when the same window is replayed twice.
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

REQUEST_COLUMNS = ["trans_num", "cc_num", "trans_ts", "amt", "category", "dob", "city_pop",
                   "lat", "long", "merch_lat", "merch_long"]  # fmt: skip


@dataclass
class ReplaySummary:
    sent: int = 0
    scored: int = 0
    rejected_out_of_order: int = 0
    errors: int = 0
    explained: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    seconds: float = 0.0
    first_ts: str | None = None
    last_ts: str | None = None

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else float("nan")

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else float("nan")

    @property
    def rate(self) -> float:
        return self.sent / self.seconds if self.seconds else 0.0


def load_replay_rows(
    test_parquet: Path, start: dt.datetime | None = None, limit: int | None = None
) -> pd.DataFrame:
    """Test transactions in time order (optionally from ``start``), with their labels."""
    df = pd.read_parquet(test_parquet, columns=[*REQUEST_COLUMNS, "is_fraud"])
    df = df.sort_values(["trans_ts", "trans_num"], kind="stable")
    if start is not None:
        df = df[df["trans_ts"] >= pd.Timestamp(start)]
    return df.head(limit).reset_index(drop=True) if limit else df.reset_index(drop=True)


def to_payload(row: dict[str, Any]) -> dict[str, Any]:
    """A DataFrame row -> the API's JSON body (live traffic: state is updated)."""
    return {
        "trans_num": str(row["trans_num"]),
        "cc_num": int(row["cc_num"]),
        "trans_ts": pd.Timestamp(row["trans_ts"]).isoformat(),
        "amt": float(row["amt"]),
        "category": str(row["category"]),
        "dob": pd.Timestamp(row["dob"]).date().isoformat(),
        "city_pop": int(row["city_pop"]),
        "lat": float(row["lat"]),
        "long": float(row["long"]),
        "merch_lat": float(row["merch_lat"]),
        "merch_long": float(row["merch_long"]),
        "update_state": True,
    }


def paced(n: int, speed: float) -> Iterator[int]:
    """Yield 0..n-1, sleeping so that items are released at ``speed`` per second on average."""
    start = time.perf_counter()
    for i in range(n):
        delay = start + i / speed - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        yield i


def replay(
    rows: pd.DataFrame,
    url: str,
    speed: float,
    explain_every: int = 0,
    client: httpx.Client | None = None,
    progress_every: int = 500,
) -> tuple[ReplaySummary, pd.DataFrame]:
    """Send ``rows`` to the API; return a summary and one result row per transaction.

    Every ``explain_every``-th transaction goes to /predict_explained (0 = never), to exercise
    the LLM path without exhausting a free API quota.
    """
    if speed <= 0:
        raise ValueError("speed must be positive (transactions per second)")
    own_client = client is None
    client = client or httpx.Client(base_url=url.rstrip("/"), timeout=60.0)
    summary, results = ReplaySummary(), []
    records = rows.to_dict("records")
    start = time.perf_counter()
    try:
        for i in paced(len(records), speed):
            row = records[i]
            explain = explain_every > 0 and (i + 1) % explain_every == 0
            path = "/predict_explained" if explain else "/predict"
            summary.sent += 1
            out: dict[str, Any] = {"trans_num": row["trans_num"], "trans_ts": row["trans_ts"],
                                   "is_fraud": int(row["is_fraud"]), "endpoint": path}  # fmt: skip
            try:
                r = client.post(path, json=to_payload(row))
            except httpx.HTTPError as exc:
                summary.errors += 1
                results.append({**out, "status": -1, "error": str(exc)[:200]})
                continue
            out["status"] = r.status_code
            if r.status_code == 200:
                body = r.json()
                summary.scored += 1
                summary.explained += int(explain)
                flagged, fraud = bool(body["flagged"]), bool(row["is_fraud"])
                summary.tp += int(flagged and fraud)
                summary.fp += int(flagged and not fraud)
                summary.fn += int(not flagged and fraud)
                out.update(score=body["score"], flagged=flagged, latency_ms=body["latency_ms"])
            elif r.status_code == 409:
                summary.rejected_out_of_order += 1
            else:
                summary.errors += 1
                out["error"] = r.text[:200]
            results.append(out)
            if progress_every and summary.sent % progress_every == 0:
                logger.info(
                    "replayed %s/%s (up to %s): recall %.3f, precision %.3f, %.1f tx/s",
                    f"{summary.sent:,}", f"{len(records):,}", row["trans_ts"], summary.recall,
                    summary.precision, summary.sent / (time.perf_counter() - start),
                )  # fmt: skip
    finally:
        if own_client:
            client.close()
    summary.seconds = time.perf_counter() - start
    if records:
        summary.first_ts = str(records[0]["trans_ts"])
        summary.last_ts = str(records[-1]["trans_ts"])
    return summary, pd.DataFrame(results)
