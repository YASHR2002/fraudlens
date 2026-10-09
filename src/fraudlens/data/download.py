"""Get the raw dataset: detect manually placed files, or download with the Kaggle CLI."""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


class DownloadError(RuntimeError):
    """The dataset could not be found or downloaded."""


def missing_files(raw_dir: Path, filenames: list[str]) -> list[str]:
    """Return the names in ``filenames`` that are not present (or empty) in ``raw_dir``."""
    return [
        n for n in filenames if not (raw_dir / n).is_file() or (raw_dir / n).stat().st_size == 0
    ]


def _kaggle_executable() -> str:
    # Prefer the CLI installed in the active virtual environment.
    venv_bin = Path(sys.executable).parent
    found = shutil.which("kaggle", path=str(venv_bin)) or shutil.which("kaggle")
    if not found:
        raise DownloadError("The `kaggle` CLI is not installed; run `uv sync`.")
    return found


def download_dataset(
    dataset: str, raw_dir: Path, filenames: list[str], force: bool = False
) -> list[Path]:
    """Ensure the raw CSVs exist in ``raw_dir``, downloading them from Kaggle if needed.

    The Kaggle CLI is called as a subprocess rather than imported, because importing the
    ``kaggle`` package tries to authenticate immediately.

    Args:
        dataset: Kaggle dataset slug, e.g. ``kartik2112/fraud-detection``.
        raw_dir: Destination folder.
        filenames: Files that must exist afterwards.
        force: Download even if the files are already present.

    Returns:
        Paths of the required files.

    Raises:
        DownloadError: If authentication or the download fails, or files are still missing.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    missing = missing_files(raw_dir, filenames)
    if not missing and not force:
        logger.info("Found %s in %s; skipping download.", ", ".join(filenames), raw_dir)
        return [raw_dir / n for n in filenames]

    logger.info("Downloading %s from Kaggle into %s (missing: %s)", dataset, raw_dir, missing)
    cmd = [_kaggle_executable(), "datasets", "download", dataset, "-p", str(raw_dir), "--unzip"]
    if force:
        cmd.append("--force")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise DownloadError(
            "Kaggle download failed. Authenticate with `uv run kaggle auth login` or set "
            "KAGGLE_API_TOKEN (see docs/setup_windows.md), or download the files manually "
            f"into {raw_dir}.\n{(result.stderr or result.stdout).strip()[-2000:]}"
        )
    still_missing = missing_files(raw_dir, filenames)
    if still_missing:
        raise DownloadError(f"Download finished but these files are missing: {still_missing}")
    return [raw_dir / n for n in filenames]
