"""Project configuration.

Two sources, never hard-coded values:

* ``configs/config.yaml``: tunable, non-secret settings (paths, split dates, costs, rules).
* Environment variables / ``.env``: secrets and deployment-specific settings.
"""

from __future__ import annotations

import datetime as dt
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

CONFIG_ENV_VAR = "FRAUDLENS_CONFIG"
DEFAULT_CONFIG_RELPATH = Path("configs") / "config.yaml"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PathsConfig(_Strict):
    raw: Path
    processed: Path
    reference: Path
    reports: Path = Path("reports")
    models: Path = Path("models")


class DataConfig(_Strict):
    """Source dataset and file names."""

    kaggle_dataset: str
    train_file: str = "fraudTrain.csv"
    test_file: str = "fraudTest.csv"
    csv_block_size_mb: int = Field(default=32, gt=0, description="CSV read chunk size.")


class SplitConfig(_Strict):
    """Time-based split. Dates are filled in after inspecting the data (Phase 1)."""

    train_end: dt.date | None = None
    validation_end: dt.date | None = None


class CostConfig(_Strict):
    false_positive_cost: float = Field(gt=0, description="Analyst review cost per false alarm.")
    false_negative_cost: Literal["amount"] | float = Field(
        description="'amount' = a missed fraud costs its transaction amount."
    )


class TrainingConfig(_Strict):
    """MLflow experiment and per-model hyperparameters."""

    experiment_name: str = "fraudlens"
    registered_model_name: str = "fraudlens-fraud-detector"
    champion_alias: str = "champion"
    best_params_file: Path = Path("configs/best_params.json")
    models: dict[str, dict[str, Any]]


class PromotionConfig(_Strict):
    min_recall: float = Field(ge=0, le=1)
    min_precision: float = Field(ge=0, le=1)
    require_pr_auc_at_least_champion: bool = True


class ShapConfig(_Strict):
    global_sample_size: int = Field(gt=0)


class LLMConfig(_Strict):
    provider: Literal["gemini"] = "gemini"
    model_env_var: str = "GEMINI_MODEL"
    timeout_seconds: float = Field(gt=0)
    thinking_level: Literal["low", "medium", "high"] | None = "low"
    max_attempts: int = Field(default=2, ge=1, le=3)
    top_factors: int = Field(default=5, ge=1, le=10)


class AppConfig(_Strict):
    """Validated contents of ``configs/config.yaml``."""

    seed: int
    paths: PathsConfig
    data: DataConfig
    split: SplitConfig
    costs: CostConfig
    training: TrainingConfig
    promotion: PromotionConfig
    shap: ShapConfig
    llm: LLMConfig
    project_root: Path = Field(default=Path(), exclude=True)

    def resolve(self, path: Path) -> Path:
        """Return ``path`` as an absolute path anchored at the project root."""
        return path if path.is_absolute() else (self.project_root / path).resolve()


class EnvSettings(BaseSettings):
    """Secrets and deployment settings from the environment (or a ``.env`` file)."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", env_ignore_empty=True
    )

    postgres_user: str = "fraudlens"
    postgres_password: SecretStr = SecretStr("change_me")
    postgres_db: str = "fraudlens"
    postgres_host: str = "127.0.0.1"  # not localhost: IPv6 ::1 stalls on Docker Desktop
    postgres_port: int = 5432
    mlflow_tracking_uri: str = "http://127.0.0.1:5000"
    google_api_key: SecretStr | None = None
    gemini_model: str | None = None
    fraud_threshold: float | None = Field(default=None, ge=0, le=1)
    model_source: Literal["mlflow", "huggingface", "local", "s3"] = "mlflow"
    hf_repo_id: str | None = None
    hf_token: SecretStr | None = None

    @property
    def postgres_conninfo(self) -> str:
        """libpq connection string for psycopg."""
        return (
            f"postgresql://{self.postgres_user}:"
            f"{quote(self.postgres_password.get_secret_value(), safe='')}@"
            f"{self.postgres_host}:{self.postgres_port}/{self.postgres_db}?connect_timeout=10"
        )

    @property
    def database_url(self) -> str:
        """SQLAlchemy URL for PostgreSQL (psycopg 3 driver)."""
        return self.postgres_conninfo.replace("postgresql://", "postgresql+psycopg://", 1)


def find_config_file(start: Path | None = None) -> Path:
    """Locate ``config.yaml``.

    Order: the ``FRAUDLENS_CONFIG`` env var, then ``configs/config.yaml`` in ``start``
    (default: the current directory) or any parent directory.

    Raises:
        FileNotFoundError: If no config file can be found.
    """
    if env_path := os.environ.get(CONFIG_ENV_VAR):
        path = Path(env_path)
        if not path.is_file():
            raise FileNotFoundError(f"{CONFIG_ENV_VAR}={env_path} does not exist")
        return path.resolve()
    here = (start or Path.cwd()).resolve()
    for directory in (here, *here.parents):
        candidate = directory / DEFAULT_CONFIG_RELPATH
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find {DEFAULT_CONFIG_RELPATH} in {here} or its parents; "
        f"run from the project directory or set {CONFIG_ENV_VAR}."
    )


def load_config(path: Path | None = None) -> AppConfig:
    """Load and validate the YAML config.

    Args:
        path: Explicit config path; if omitted, :func:`find_config_file` is used.
    """
    config_path = (path or find_config_file()).resolve()
    with config_path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    # configs/config.yaml lives one level below the project root.
    return AppConfig(**raw, project_root=config_path.parent.parent)


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Cached project config for application code."""
    return load_config()


@lru_cache(maxsize=1)
def get_env() -> EnvSettings:
    """Cached environment settings for application code."""
    return EnvSettings()
