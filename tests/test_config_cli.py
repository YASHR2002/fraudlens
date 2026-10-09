"""Smoke tests for configuration loading and the CLI entry point."""

from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from fraudlens import __version__
from fraudlens.cli import app
from fraudlens.config import AppConfig, EnvSettings, find_config_file, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()


def test_project_config_is_valid() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "config.yaml")
    assert config.seed == 42
    assert config.costs.false_negative_cost == "amount"
    assert config.project_root == PROJECT_ROOT
    assert config.resolve(config.paths.raw) == PROJECT_ROOT / "data" / "raw"


def test_find_config_file_walks_up(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FRAUDLENS_CONFIG", raising=False)
    found = find_config_file(start=PROJECT_ROOT / "src" / "fraudlens")
    assert found == PROJECT_ROOT / "configs" / "config.yaml"


def test_unknown_config_key_is_rejected() -> None:
    raw = load_config(PROJECT_ROOT / "configs" / "config.yaml").model_dump()
    raw["typo_key"] = 1
    with pytest.raises(ValidationError):
        AppConfig(**raw)


def test_secrets_are_masked_in_serialised_settings() -> None:
    env = EnvSettings(_env_file=None, postgres_password="s3cret", google_api_key="key123")
    dumped = env.model_dump_json()
    assert "s3cret" not in dumped
    assert "key123" not in dumped
    assert "s3cret" in env.database_url


def test_cli_help_and_version() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "show-config" in result.output

    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output
