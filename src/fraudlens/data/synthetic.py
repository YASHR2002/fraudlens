"""Generate a small synthetic transactions file in the raw Sparkov CSV format.

Used as the test fixture (``tests/fixtures/sample_transactions.csv``): every value is
invented here, nothing is copied from the real dataset. It deliberately includes the
quirks the pipeline must handle: an unnamed index column, 4-digit zips, several
transactions per card (for velocity features later), and a few fraud rows.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from fraudlens.data.schema import EXPECTED_HEADER, RAW_INDEX_COLUMN

CATEGORIES = [
    "grocery_pos", "gas_transport", "shopping_net", "misc_net", "food_dining",
    "entertainment", "health_fitness", "travel",
]  # fmt: skip
MERCHANTS = [f"fraud_Test Merchant {i}" for i in range(1, 16)]
JOBS = ["Engineer, test", "Teacher, sample", "Analyst, synthetic"]


def _cards(rng: np.random.Generator, n_cards: int) -> list[dict[str, object]]:
    cards = []
    for i in range(n_cards):
        lat = float(rng.uniform(30, 45))
        lon = float(rng.uniform(-120, -75))
        cards.append(
            {
                "cc_num": int(4_000_000_000_000_000 + i * 1_111_111),
                "first": f"First{i}",
                "last": f"Last{i}",
                "gender": "F" if i % 2 == 0 else "M",
                "street": f"{100 + i} Test Street",
                "city": f"Testville {i}",
                "state": ["CA", "NY", "TX", "MA", "NJ", "WA"][i % 6],
                # MA/NJ-style zips lose their leading zero in the real CSVs; mimic that.
                "zip": str(rng.integers(1000, 9999))
                if i % 3 == 0
                else str(rng.integers(10000, 99999)),
                "lat": round(lat, 4),
                "long": round(lon, 4),
                "city_pop": int(rng.integers(100, 2_000_000)),
                "job": JOBS[i % len(JOBS)],
                "dob": dt.date(1950 + 7 * i, 1 + i % 12, 1 + i % 28),
            }
        )
    return cards


def generate_sample_transactions(
    n_rows: int = 200, n_cards: int = 8, n_fraud: int = 8, seed: int = 42
) -> pd.DataFrame:
    """Return a synthetic raw-format DataFrame sorted by time, with a few fraud rows."""
    rng = np.random.default_rng(seed)
    cards = _cards(rng, n_cards)
    start = dt.datetime(2019, 1, 1)
    # Random gaps of 0-12 hours, so cards get both bursts and quiet periods.
    offsets = np.cumsum(rng.integers(1, 12 * 3600, size=n_rows))
    rows = []
    for k, offset in enumerate(offsets):
        card = cards[int(rng.integers(0, n_cards))]
        ts = start + dt.timedelta(seconds=int(offset))
        rows.append(
            {
                "trans_date_trans_time": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "cc_num": card["cc_num"],
                "merchant": MERCHANTS[int(rng.integers(0, len(MERCHANTS)))],
                "category": CATEGORIES[int(rng.integers(0, len(CATEGORIES)))],
                "amt": round(float(rng.lognormal(3.5, 1.0)) + 1.0, 2),
                **{f: card[f] for f in ("first", "last", "gender", "street", "city", "state")},
                "zip": card["zip"],
                "lat": card["lat"],
                "long": card["long"],
                "city_pop": card["city_pop"],
                "job": card["job"],
                "dob": card["dob"].isoformat(),
                "trans_num": rng.bytes(16).hex(),
                "unix_time": int(ts.replace(tzinfo=dt.UTC).timestamp()),
                "merch_lat": round(float(card["lat"]) + float(rng.normal(0, 0.5)), 6),
                "merch_long": round(float(card["long"]) + float(rng.normal(0, 0.5)), 6),
                "is_fraud": 0,
                "_k": k,
            }
        )
    df = pd.DataFrame(rows).drop(columns="_k")
    # Fraud rows: large online purchases, so they differ from the card's usual spend.
    n_base = n_rows - len(EDGE_CASE_GAPS_S)  # the rest are edge-case copies, added below
    fraud_idx = rng.choice(np.arange(10, n_base), size=n_fraud, replace=False)
    df.loc[fraud_idx, "is_fraud"] = 1
    df.loc[fraud_idx, "amt"] = np.round(rng.uniform(300, 1200, size=n_fraud), 2)
    df.loc[fraud_idx, "category"] = "shopping_net"
    df = _add_edge_cases(df, rng, n_rows)
    df.insert(0, RAW_INDEX_COLUMN, range(len(df)))
    return df[EXPECTED_HEADER]


# Extra transactions placed exactly on feature-window edges (all occur in the real data):
# same-second ties on a card, and gaps of exactly 1 hour, 24 hours and 7 days.
EDGE_CASE_GAPS_S = (0, 0, 0, 3_600, 3_600, 3_600, 86_400, 86_400, 604_800)


def _add_edge_cases(df: pd.DataFrame, rng: np.random.Generator, n_rows: int) -> pd.DataFrame:
    base = df.iloc[: n_rows - len(EDGE_CASE_GAPS_S)].copy()
    sources = rng.choice(np.arange(len(base)), size=len(EDGE_CASE_GAPS_S), replace=False)
    extras = []
    for src, gap in zip(sources, EDGE_CASE_GAPS_S, strict=True):
        row = base.iloc[int(src)].copy()
        ts = dt.datetime.fromisoformat(row["trans_date_trans_time"]) + dt.timedelta(seconds=gap)
        row["trans_date_trans_time"] = ts.strftime("%Y-%m-%d %H:%M:%S")
        row["unix_time"] = int(ts.replace(tzinfo=dt.UTC).timestamp())
        row["trans_num"] = rng.bytes(16).hex()
        row["amt"] = round(float(rng.lognormal(3.5, 1.0)) + 1.0, 2)
        row["category"] = CATEGORIES[int(rng.integers(0, len(CATEGORIES)))]
        row["is_fraud"] = 0
        extras.append(row)
    out = pd.concat([base, pd.DataFrame(extras)], ignore_index=True)
    return out.sort_values("trans_date_trans_time", kind="stable").reset_index(drop=True)


def write_sample_csv(path: Path, **kwargs: int) -> Path:
    """Generate the synthetic sample and write it as a raw-format CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    generate_sample_transactions(**kwargs).to_csv(path, index=False, lineterminator="\n")
    return path
