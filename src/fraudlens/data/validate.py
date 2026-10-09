"""Data quality checks on the raw CSVs, with a Markdown report.

One streaming pass per file collects statistics; rules are then evaluated on them.
Schema problems raise immediately (:class:`~fraudlens.data.io.SchemaError`); rule
violations are written to the report and raise :class:`DataQualityError` afterwards.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pyarrow as pa
import pyarrow.compute as pc

from fraudlens.data.io import iter_raw_batches
from fraudlens.data.schema import LAT_RANGE, LONG_RANGE, RAW_COLUMNS, VALID_GENDERS

logger = logging.getLogger(__name__)

Severity = Literal["error", "warning", "info"]


class DataQualityError(RuntimeError):
    """At least one error-level data quality check failed."""


@dataclass
class FileProfile:
    """Statistics collected from one raw file in a single pass."""

    name: str
    rows: int = 0
    null_counts: Counter[str] = field(default_factory=Counter)
    duplicate_trans_num: int = 0
    non_positive_amt: int = 0
    amt_min: float = float("inf")
    amt_max: float = float("-inf")
    bad_coordinates: int = 0
    bad_gender: int = 0
    bad_is_fraud: int = 0
    bad_age: int = 0
    short_zip: int = 0
    out_of_order: int = 0
    fraud: int = 0
    ts_min: dt.datetime | None = None
    ts_max: dt.datetime | None = None
    unix_offset_min: int | None = None
    unix_offset_max: int | None = None
    cards: set[int] = field(default_factory=set)
    trans_nums: set[str] = field(default_factory=set)
    monthly: dict[str, list[int]] = field(default_factory=dict)  # month -> [txns, fraud]

    @property
    def fraud_rate(self) -> float:
        return self.fraud / self.rows if self.rows else 0.0


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one rule."""

    name: str
    severity: Severity
    passed: bool
    detail: str


def _count_outside(arr: pa.ChunkedArray | pa.Array, lo: float, hi: float) -> int:
    inside = pc.and_(pc.greater_equal(arr, lo), pc.less_equal(arr, hi))
    return int(pc.sum(pc.invert(inside)).as_py() or 0)


def _update_minmax(current: object, new: object, pick: type[min] | type[max]) -> object:
    return new if current is None else pick(current, new)


def profile_file(path: Path, block_size_mb: int = 32) -> FileProfile:
    """Stream ``path`` once and collect the statistics needed for validation."""
    prof = FileProfile(name=path.name)
    last_ts: int | None = None
    for batch in iter_raw_batches(path, block_size_mb):
        n = batch.num_rows
        prof.rows += n
        for name in RAW_COLUMNS:
            prof.null_counts[name] += batch.column(name).null_count

        ts = batch.column("trans_date_trans_time")
        ts_int = pc.cast(ts, pa.int64())
        # Rows out of time order (within the batch, and across the batch boundary).
        diffs = pc.pairwise_diff(ts_int)
        prof.out_of_order += int(pc.sum(pc.less(diffs, 0)).as_py() or 0)
        first_ts = ts_int[0].as_py()
        if last_ts is not None and first_ts is not None and first_ts < last_ts:
            prof.out_of_order += 1
        last_ts = ts_int[n - 1].as_py()
        mm = pc.min_max(ts)
        prof.ts_min = _update_minmax(prof.ts_min, mm["min"].as_py(), min)
        prof.ts_max = _update_minmax(prof.ts_max, mm["max"].as_py(), max)

        # unix_time should agree with the timestamp; record the offset between them.
        offset = pc.subtract(batch.column("unix_time"), ts_int)
        omm = pc.min_max(offset)
        prof.unix_offset_min = _update_minmax(prof.unix_offset_min, omm["min"].as_py(), min)
        prof.unix_offset_max = _update_minmax(prof.unix_offset_max, omm["max"].as_py(), max)

        amt = batch.column("amt")
        prof.non_positive_amt += int(pc.sum(pc.less_equal(amt, 0)).as_py() or 0)
        amm = pc.min_max(amt)
        prof.amt_min = min(prof.amt_min, amm["min"].as_py())
        prof.amt_max = max(prof.amt_max, amm["max"].as_py())

        for col in ("lat", "merch_lat"):
            prof.bad_coordinates += _count_outside(batch.column(col), *LAT_RANGE)
        for col in ("long", "merch_long"):
            prof.bad_coordinates += _count_outside(batch.column(col), *LONG_RANGE)

        valid_gender = pc.is_in(batch.column("gender"), pa.array(sorted(VALID_GENDERS)))
        prof.bad_gender += int(pc.sum(pc.invert(valid_gender)).as_py() or 0)

        is_fraud = batch.column("is_fraud")
        prof.bad_is_fraud += int(pc.sum(pc.invert(pc.is_in(is_fraud, pa.array([0, 1])))).as_py())
        prof.fraud += int(pc.sum(is_fraud).as_py() or 0)

        dob = pc.cast(batch.column("dob"), pa.timestamp("s"))
        age_years = pc.divide(pc.cast(pc.days_between(dob, ts), pa.float64()), 365.25)
        prof.bad_age += _count_outside(age_years, 0, 120)

        prof.short_zip += int(pc.sum(pc.less(pc.utf8_length(batch.column("zip")), 5)).as_py())

        prof.cards.update(pc.unique(batch.column("cc_num")).to_pylist())
        trans_nums = batch.column("trans_num").to_pylist()
        before = len(prof.trans_nums)
        prof.trans_nums.update(trans_nums)
        prof.duplicate_trans_num += n - (len(prof.trans_nums) - before)

        months = pc.strftime(ts, format="%Y-%m")
        grouped = (
            pa.table({"month": months, "is_fraud": is_fraud})
            .group_by("month")
            .aggregate([("is_fraud", "count"), ("is_fraud", "sum")])
        )
        for month, cnt, fr in zip(
            grouped["month"].to_pylist(),
            grouped["is_fraud_count"].to_pylist(),
            grouped["is_fraud_sum"].to_pylist(),
            strict=True,
        ):
            acc = prof.monthly.setdefault(month, [0, 0])
            acc[0] += cnt
            acc[1] += fr
    logger.info("Profiled %s: %s rows, %s fraud", path.name, f"{prof.rows:,}", f"{prof.fraud:,}")
    return prof


def run_checks(train: FileProfile, test: FileProfile) -> list[CheckResult]:
    """Evaluate the data quality rules on both file profiles."""
    results: list[CheckResult] = []

    def add(name: str, severity: Severity, passed: bool, detail: str) -> None:
        results.append(CheckResult(name, severity, passed, detail))

    for p in (train, test):
        nulls = {k: v for k, v in p.null_counts.items() if v}
        add(f"{p.name}: no nulls", "error", not nulls, str(nulls) if nulls else "0 nulls")
        add(
            f"{p.name}: trans_num unique",
            "error",
            p.duplicate_trans_num == 0,
            f"{p.duplicate_trans_num:,} duplicates",
        )
        add(
            f"{p.name}: amount > 0",
            "error",
            p.non_positive_amt == 0,
            f"{p.non_positive_amt:,} rows <= 0; range {p.amt_min:,.2f} to {p.amt_max:,.2f}",
        )
        add(
            f"{p.name}: valid lat/long",
            "error",
            p.bad_coordinates == 0,
            f"{p.bad_coordinates:,} out-of-range values",
        )
        add(
            f"{p.name}: is_fraud in {{0, 1}}",
            "error",
            p.bad_is_fraud == 0,
            f"{p.bad_is_fraud:,} invalid labels",
        )
        add(
            f"{p.name}: gender in {{F, M}}",
            "error",
            p.bad_gender == 0,
            f"{p.bad_gender:,} other values",
        )
        add(
            f"{p.name}: age at transaction 0-120",
            "error",
            p.bad_age == 0,
            f"{p.bad_age:,} rows outside range",
        )
        add(
            f"{p.name}: has fraud and legit rows",
            "error",
            0 < p.fraud < p.rows,
            f"{p.fraud:,} fraud ({p.fraud_rate:.3%})",
        )
        add(
            f"{p.name}: sorted by time",
            "warning",
            p.out_of_order == 0,
            f"{p.out_of_order:,} rows earlier than the previous row",
        )
        add(
            f"{p.name}: 5-digit zip codes",
            "warning",
            p.short_zip == 0,
            f"{p.short_zip:,} zips shorter than 5 digits (leading zeros lost; padded on convert)",
        )
        consistent = p.unix_offset_min == p.unix_offset_max == 0
        add(
            f"{p.name}: unix_time matches timestamp",
            "warning",
            consistent,
            f"unix_time - timestamp ranges {p.unix_offset_min:,} to {p.unix_offset_max:,} s "
            "(about {:.2f} years); trans_date_trans_time is used as the event time".format(
                (p.unix_offset_min or 0) / (365.25 * 86400)
            ),
        )

    overlap = len(train.trans_nums & test.trans_nums)
    add("train/test: no shared trans_num", "error", overlap == 0, f"{overlap:,} shared")
    in_order = train.ts_max is not None and test.ts_min is not None and test.ts_min > train.ts_max
    add(
        "train/test: test starts after train ends",
        "error",
        in_order,
        f"train ends {train.ts_max}, test starts {test.ts_min}",
    )
    new_cards = len(test.cards - train.cards)
    add(
        "train/test: cards unseen in train",
        "info",
        True,
        f"{new_cards:,} of {len(test.cards):,} test cards have no history in fraudTrain",
    )
    return results


def _fmt_ts(value: dt.datetime | None) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S") if value else "n/a"


def render_report(train: FileProfile, test: FileProfile, checks: list[CheckResult]) -> str:
    """Render the data quality report as Markdown."""
    errors = [c for c in checks if c.severity == "error" and not c.passed]
    status = "FAILED" if errors else "PASSED"
    lines = [
        "# Data Quality Report",
        "",
        f"Generated {dt.datetime.now():%Y-%m-%d %H:%M} by `fraudlens validate-data`. "
        f"Overall status: **{status}** ({len(errors)} error-level failures).",
        "",
        "## Summary",
        "",
        "| | " + " | ".join(p.name for p in (train, test)) + " |",
        "|---|---:|---:|",
    ]
    rows = [
        ("Rows", lambda p: f"{p.rows:,}"),
        ("Fraud rows", lambda p: f"{p.fraud:,}"),
        ("Fraud rate", lambda p: f"{p.fraud_rate:.3%}"),
        ("First transaction", lambda p: _fmt_ts(p.ts_min)),
        ("Last transaction", lambda p: _fmt_ts(p.ts_max)),
        ("Distinct cards", lambda p: f"{len(p.cards):,}"),
        ("Amount range (USD)", lambda p: f"{p.amt_min:,.2f} to {p.amt_max:,.2f}"),
    ]
    for label, fn in rows:
        lines.append(f"| {label} | {fn(train)} | {fn(test)} |")

    lines += ["", "## Checks", "", "| Check | Severity | Result | Detail |", "|---|---|---|---|"]
    for c in checks:
        result = "pass" if c.passed else ("FAIL" if c.severity == "error" else "flag")
        lines.append(f"| {c.name} | {c.severity} | {result} | {c.detail} |")

    lines += [
        "",
        "## Transactions and fraud by month",
        "",
        "| Month | File | Transactions | Fraud | Fraud rate |",
        "|---|---|---:|---:|---:|",
    ]
    for p in (train, test):
        for month in sorted(p.monthly):
            n, f = p.monthly[month]
            lines.append(f"| {month} | {p.name} | {n:,} | {f:,} | {f / n:.3%} |")
    lines.append("")
    return "\n".join(lines)


def validate_data(
    train_path: Path, test_path: Path, report_path: Path, block_size_mb: int = 32
) -> list[CheckResult]:
    """Profile both raw files, write the Markdown report, and fail on error-level checks.

    Raises:
        SchemaError: If a file's columns or types do not match the expected schema.
        DataQualityError: If any error-level check fails (after the report is written).
    """
    train = profile_file(train_path, block_size_mb)
    test = profile_file(test_path, block_size_mb)
    checks = run_checks(train, test)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(train, test, checks), encoding="utf-8")
    logger.info("Wrote %s", report_path)

    for c in checks:
        if not c.passed:
            log = logger.error if c.severity == "error" else logger.warning
            log("%s check '%s' failed: %s", c.severity.upper(), c.name, c.detail)
    errors = [c for c in checks if c.severity == "error" and not c.passed]
    if errors:
        raise DataQualityError(
            f"{len(errors)} data quality error(s); see {report_path}: "
            + "; ".join(c.name for c in errors)
        )
    return checks
