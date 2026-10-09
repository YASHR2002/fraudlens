"""Tests for Hugging Face publishing and loading (the Hub API is mocked; no network)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import huggingface_hub
import pytest
import yaml

from fraudlens.api.model_loader import ModelLoadError, load_from_huggingface
from fraudlens.config import EnvSettings
from fraudlens.deploy import hub

ROOT = Path(__file__).parents[1]
META = {"name": "fraudlens-fraud-detector", "version": "3", "family": "lightgbm",
        "threshold": 0.31, "metrics": {"test_pr_auc": 0.97, "test_recall": 0.95}}  # fmt: skip


@pytest.fixture
def exported(tmp_path: Path) -> tuple[Path, Path]:
    champion = tmp_path / "models" / "champion"
    (champion / "model").mkdir(parents=True)
    (champion / "model" / "model.skops").write_bytes(b"x")
    (champion / "metadata.json").write_text(json.dumps(META), encoding="utf-8")
    processed = tmp_path / "processed"
    processed.mkdir()
    for f in hub.STATE_FILES:
        (processed / f).write_bytes(b"x")
    return processed, champion


@pytest.fixture
def fake_api(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    api = MagicMock()
    api.upload_folder.return_value = MagicMock(commit_url="https://hf.co/commit/1")
    monkeypatch.setattr(huggingface_hub, "HfApi", MagicMock(return_value=api))
    return api


def test_model_card_handles_missing_metrics() -> None:
    card = hub.model_card(META, "u/m", "https://github.com/x/y")
    assert card.startswith("---\n") and "97.0%" not in card  # recall formatted, pr-auc 4 dp
    assert "0.9700" in card and "95.0%" in card and "n/a" in card and "0.3100" in card


def test_stage_model_layout(exported: tuple[Path, Path]) -> None:
    stage = hub.stage_model(*exported, "https://github.com/x/y", "u/m")
    files = {p.relative_to(stage).as_posix() for p in stage.rglob("*") if p.is_file()}
    assert files == {"README.md", "features.json", "champion/metadata.json",
                     "champion/model/model.skops", "state/card_state.json.gz",
                     "state/demo_transactions.parquet"}  # fmt: skip
    assert len(json.loads((stage / "features.json").read_text(encoding="utf-8"))) == 18


def test_stage_model_requires_export_and_state(exported: tuple[Path, Path]) -> None:
    processed, champion = exported
    (processed / "card_state.json.gz").unlink()
    with pytest.raises(FileNotFoundError, match="build-state"):
        hub.stage_model(processed, champion, "g", "u/m")
    with pytest.raises(FileNotFoundError, match="export-model"):
        hub.stage_model(processed, champion.parent / "missing", "g", "u/m")


def test_push_model_uploads_and_cleans_up(exported, fake_api: MagicMock) -> None:
    url = hub.push_model(*exported, "u/m", "tok", "g", private=True)
    assert url == "https://hf.co/commit/1"
    fake_api.create_repo.assert_called_once_with("u/m", repo_type="model", private=True,
                                                 exist_ok=True)  # fmt: skip
    kwargs = fake_api.upload_folder.call_args.kwargs
    assert kwargs["repo_id"] == "u/m" and "v3 (lightgbm)" in kwargs["commit_message"]
    assert not Path(kwargs["folder_path"]).exists()  # staging folder removed


def test_publish_space(fake_api: MagicMock) -> None:
    dashboard = ROOT / "src" / "fraudlens" / "dashboard" / "app.py"
    seen: dict[str, str] = {}

    def capture(**kwargs):
        folder = Path(kwargs["folder_path"])
        seen.update({p.name: p.read_text(encoding="utf-8") for p in folder.iterdir()})
        return MagicMock(commit_url="c")

    fake_api.upload_folder.side_effect = capture
    hub.publish_space(dashboard, "u/fraudlens", "https://api.example.com", "tok", "g")
    assert set(seen) == {"app.py", "Dockerfile", "requirements.txt", "README.md"}
    front = yaml.safe_load(seen["README.md"].split("---")[1])
    assert front["sdk"] == "docker" and front["app_port"] == 8501
    assert {line.split("==")[0] for line in seen["requirements.txt"].split()} == {
        "streamlit", "plotly", "httpx"}  # fmt: skip
    assert "8501" in seen["Dockerfile"] and "--uid 1000" in seen["Dockerfile"]
    fake_api.create_repo.assert_called_once_with("u/fraudlens", repo_type="space",
                                                 space_sdk="docker", exist_ok=True)  # fmt: skip
    fake_api.add_space_variable.assert_called_once()
    assert fake_api.add_space_variable.call_args.args[1:] == ("API_URL", "https://api.example.com")


def test_huggingface_source_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ModelLoadError, match="needs HF_REPO_ID"):
        load_from_huggingface(EnvSettings(_env_file=None, model_source="huggingface"))

    def offline(**kwargs):
        raise OSError("no network")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", offline)
    env = EnvSettings(_env_file=None, model_source="huggingface", hf_repo_id="u/m")
    with pytest.raises(ModelLoadError, match="cannot download u/m"):
        load_from_huggingface(env)


def test_render_blueprint_matches_the_app() -> None:
    spec = yaml.safe_load((ROOT / "render.yaml").read_text(encoding="utf-8"))
    (svc,) = spec["services"]
    assert svc["plan"] == "free" and svc["runtime"] == "docker"
    assert (ROOT / svc["dockerfilePath"]).is_file() and svc["healthCheckPath"] == "/health"
    env = {e["key"]: e for e in svc["envVars"]}
    assert env["MODEL_SOURCE"]["value"] == "huggingface"
    assert all(env[k]["sync"] is False for k in ("HF_TOKEN", "GOOGLE_API_KEY", "HF_REPO_ID"))
    assert "${PORT:-8000}" in (ROOT / "docker" / "api.Dockerfile").read_text(encoding="utf-8")


def test_cli_commands(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from fraudlens.cli import app
    from fraudlens.config import get_env

    runner = CliRunner()
    monkeypatch.chdir(tmp_path)  # no .env here
    monkeypatch.setenv("FRAUDLENS_CONFIG", str(ROOT / "configs" / "config.yaml"))
    for key in ("HF_TOKEN", "HF_REPO_ID"):
        monkeypatch.delenv(key, raising=False)
    get_env.cache_clear()
    assert "--repo-id or set HF_REPO_ID" in runner.invoke(app, ["push-model"]).output
    assert "HF_TOKEN is not set" in runner.invoke(app, ["push-model", "--repo-id", "u/m"]).output

    monkeypatch.setenv("HF_TOKEN", "tok")
    get_env.cache_clear()
    pushed: list[tuple] = []
    monkeypatch.setattr(hub, "push_model", lambda *a: pushed.append(a) or "c1")
    monkeypatch.setattr(hub, "publish_space", lambda *a: pushed.append(a) or "c2")
    out = runner.invoke(app, ["push-model", "--repo-id", "u/m"], catch_exceptions=False).output
    assert "huggingface.co/u/m" in out and pushed[0][2:4] == ("u/m", "tok")
    out = runner.invoke(app, ["publish-space", "--space-id", "u/s", "--api-url", "https://a/"],
                        catch_exceptions=False).output  # fmt: skip
    assert "spaces/u/s" in out and pushed[1][1:3] == ("u/s", "https://a")
    get_env.cache_clear()
