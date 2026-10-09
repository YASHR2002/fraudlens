"""Small helpers: dataset download (subprocess mocked) and timing logs."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from fraudlens.data import download
from fraudlens.data.download import DownloadError, download_dataset
from fraudlens.logging_utils import log_duration, setup_logging

FILES = ["fraudTrain.csv", "fraudTest.csv"]


def test_existing_files_skip_the_download(tmp_path: Path, monkeypatch) -> None:
    for f in FILES:
        (tmp_path / f).write_text("x", encoding="utf-8")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("must not download"))
    assert download_dataset("owner/data", tmp_path, FILES) == [tmp_path / f for f in FILES]


def test_download_success(tmp_path: Path, monkeypatch) -> None:
    def fake_run(cmd, **kwargs):
        assert cmd[1:4] == ["datasets", "download", "owner/data"] and "--unzip" in cmd
        for f in FILES:
            (tmp_path / f).write_text("x", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(download, "_kaggle_executable", lambda: "kaggle")
    monkeypatch.setattr(subprocess, "run", fake_run)
    assert len(download_dataset("owner/data", tmp_path, FILES)) == 2


def test_download_failure_explains_how_to_authenticate(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(download, "_kaggle_executable", lambda: "kaggle")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=1, stdout="", stderr="401 Unauthorized"))  # fmt: skip
    with pytest.raises(DownloadError, match="kaggle auth login") as err:
        download_dataset("owner/data", tmp_path, FILES)
    assert "401 Unauthorized" in str(err.value)  # the CLI's own message is passed on


def test_download_that_leaves_files_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(download, "_kaggle_executable", lambda: "kaggle")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout="", stderr=""))  # fmt: skip
    with pytest.raises(DownloadError, match="missing"):
        download_dataset("owner/data", tmp_path, FILES)


def test_log_duration(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="fraudlens"), log_duration("unit of work"):
        pass
    assert "Started: unit of work" in caplog.text and "Finished: unit of work in" in caplog.text
    with pytest.raises(RuntimeError), log_duration("failing work"):
        raise RuntimeError("boom")
    assert "Failed: failing work" in caplog.text


def test_setup_logging_configures_root_and_quietens_libraries() -> None:
    root = logging.getLogger()
    saved = (root.handlers[:], root.level)
    try:
        setup_logging("DEBUG")
        assert root.level == logging.DEBUG and root.handlers
        assert logging.getLogger("urllib3").level == logging.WARNING
    finally:
        root.handlers[:], _ = saved
        root.setLevel(saved[1])
