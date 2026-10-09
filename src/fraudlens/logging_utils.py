"""Logging setup and timing helpers shared by the CLI and library code."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str | int = "INFO") -> None:
    """Configure root logging once for CLI usage.

    Args:
        level: Logging level name (e.g. ``"DEBUG"``) or numeric level.
    """
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt=DATE_FORMAT, force=True)
    # Third-party libraries are noisy at INFO; keep them quiet unless debugging.
    for noisy in ("urllib3", "httpx", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@contextmanager
def log_duration(task: str, logger: logging.Logger | None = None) -> Iterator[None]:
    """Log the start, end, and wall-clock duration of a block of work.

    Args:
        task: Human-readable description of the work being timed.
        logger: Logger to use; defaults to the ``fraudlens`` logger.
    """
    log = logger or logging.getLogger("fraudlens")
    log.info("Started: %s", task)
    start = time.perf_counter()
    try:
        yield
    except Exception:
        log.error("Failed: %s after %.2fs", task, time.perf_counter() - start)
        raise
    log.info("Finished: %s in %.2fs", task, time.perf_counter() - start)
