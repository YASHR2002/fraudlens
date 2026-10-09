"""Publish the demo to the Hugging Face Hub.

* :func:`push_model` uploads a **model repo**: the exported champion (skops model + metadata with
  threshold and metrics), the feature list, the card-state snapshot and the demo transactions,
  plus a model card. The Render API downloads it at startup (``MODEL_SOURCE=huggingface``).
* :func:`publish_space` uploads a **Docker Space** running the Streamlit dashboard, pointed at the
  public API through the ``API_URL`` Space variable.

Files are staged in a temporary folder and uploaded in one commit. Nothing secret is uploaded:
the token is only used to authenticate.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from importlib.metadata import version
from pathlib import Path
from typing import Any

from fraudlens.api.model_loader import HF_MODEL_DIR, HF_STATE_DIR, LOCAL_METADATA
from fraudlens.features.feature_names import FEATURES

STATE_FILES = ("card_state.json.gz", "demo_transactions.parquet")
SPACE_REQUIREMENTS = ("streamlit", "plotly", "httpx")


def model_card(meta: dict[str, Any], repo_id: str, github_url: str) -> str:
    """README.md for the model repo (Hugging Face model card with YAML metadata)."""
    m = meta.get("metrics", {})

    def pct(key: str) -> str:
        return f"{m[key]:.1%}" if key in m else "n/a"

    def money(key: str) -> str:
        return f"${m[key]:,.0f}" if key in m else "n/a"

    return f"""---
tags:
- fraud-detection
- tabular-classification
- lightgbm
- shap
library_name: sklearn
---

# FraudLens fraud detector ({meta["family"]}, version {meta["version"]})

Credit card fraud scoring model from **FraudLens** ({github_url}): leakage-free behavioural
features, a cost-based decision threshold, SHAP explanations and LLM analyst notes.
This repo is what the public demo API downloads at startup.

| | Test set (fraudTest, used once) | Validation |
|---|---|---|
| PR-AUC | {m.get("test_pr_auc", float("nan")):.4f} | {m.get("val_pr_auc", float("nan")):.4f} |
| Recall at threshold | {pct("test_recall")} | {pct("val_recall")} |
| Precision at threshold | {pct("test_precision")} | {pct("val_precision")} |
| Total cost (missed fraud + $5 per false alarm) | {money("test_cost")} | {money("val_cost")} |

Decision threshold: **{meta["threshold"]:.4f}** (minimises total cost on validation).

## Contents

- `{HF_MODEL_DIR}/model/`: scikit-learn pipeline in MLflow format, serialised with **skops**
  (load with the trusted types listed in `fraudlens.models.estimators.SKOPS_TRUSTED_TYPES`).
- `{HF_MODEL_DIR}/{LOCAL_METADATA}`: version, threshold, metrics, feature importance, fairness.
- `features.json`: the 18 input features with plain-English descriptions.
- `{HF_STATE_DIR}/`: per-card state snapshot (end of the training data) and demo transactions.

## Data and limitations

Trained on the synthetic Sparkov dataset (CC0), so scores are higher than real fraud detection
would achieve; card numbers and personal fields in the demo files are synthetic. Gender is not a
model input; age is, and fairness gaps by gender and age are reported in the project's model card.
Not for production use.
"""


def stage_model(processed: Path, champion: Path, github_url: str, repo_id: str) -> Path:
    """Assemble the model repo contents in a temporary folder."""
    meta_path = champion / LOCAL_METADATA
    if not meta_path.is_file():
        raise FileNotFoundError(f"{meta_path} not found; run `fraudlens export-model` first")
    missing = [f for f in STATE_FILES if not (processed / f).is_file()]
    if missing:
        raise FileNotFoundError(f"missing {missing} in {processed}; run `fraudlens build-state`")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    stage = Path(tempfile.mkdtemp(prefix="fraudlens-model-"))
    shutil.copytree(champion, stage / HF_MODEL_DIR)
    (stage / HF_STATE_DIR).mkdir()
    for f in STATE_FILES:
        shutil.copy2(processed / f, stage / HF_STATE_DIR / f)
    features = [{"name": f.name, "label": f.label, "description": f.description, "unit": f.unit}
                for f in FEATURES]  # fmt: skip
    (stage / "features.json").write_text(json.dumps(features, indent=2), encoding="utf-8")
    (stage / "README.md").write_text(model_card(meta, repo_id, github_url), encoding="utf-8")
    return stage


def push_model(
    processed: Path, champion: Path, repo_id: str, token: str, github_url: str, private: bool
) -> str:
    """Create (if needed) and upload the model repo. Returns the commit URL."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    stage = stage_model(processed, champion, github_url, repo_id)
    try:
        api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
        meta = json.loads((stage / HF_MODEL_DIR / LOCAL_METADATA).read_text(encoding="utf-8"))
        info = api.upload_folder(
            repo_id=repo_id, folder_path=str(stage), repo_type="model",
            commit_message=f"FraudLens {meta['name']} v{meta['version']} ({meta['family']})",
        )  # fmt: skip
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return str(getattr(info, "commit_url", info))


SPACE_DOCKERFILE = """\
# FraudLens dashboard on Hugging Face Spaces (Docker SDK). Talks to the API at $API_URL.
FROM python:3.11-slim
RUN useradd --create-home --uid 1000 user
WORKDIR /home/user/app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user app.py .
USER user
ENV PYTHONUNBUFFERED=1 STREAMLIT_SERVER_HEADLESS=true STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
"""


def space_readme(api_url: str, github_url: str) -> str:
    return f"""---
title: FraudLens
emoji: 🔍
colorFrom: blue
colorTo: red
sdk: docker
app_port: 8501
pinned: true
short_description: Explainable credit card fraud scoring with SHAP and LLM notes
---

# FraudLens: explainable fraud detection demo

Pick a demo card transaction, score it with the fraud model and read why: a SHAP waterfall of the
factors and an analyst note written by an LLM from those factors only.

- Scoring API: {api_url}/docs (Render free tier: the first request after a quiet spell can take a
  minute or two while it wakes up)
- Code, evaluation and design decisions: {github_url}

Synthetic data (Sparkov); a portfolio project, not a production fraud system.
"""


def stage_space(dashboard_app: Path, api_url: str, github_url: str) -> Path:
    """Assemble the Space contents (dashboard script, pinned requirements, Dockerfile, card)."""
    stage = Path(tempfile.mkdtemp(prefix="fraudlens-space-"))
    shutil.copy2(dashboard_app, stage / "app.py")
    reqs = "\n".join(f"{pkg}=={version(pkg)}" for pkg in SPACE_REQUIREMENTS)
    (stage / "requirements.txt").write_text(reqs + "\n", encoding="utf-8")
    (stage / "Dockerfile").write_text(SPACE_DOCKERFILE, encoding="utf-8")
    (stage / "README.md").write_text(space_readme(api_url, github_url), encoding="utf-8")
    return stage


def publish_space(dashboard_app: Path, space_id: str, api_url: str, token: str,
                  github_url: str) -> str:  # fmt: skip
    """Create (if needed) and upload the dashboard Space; set its API_URL variable."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    stage = stage_space(dashboard_app, api_url, github_url)
    try:
        api.create_repo(space_id, repo_type="space", space_sdk="docker", exist_ok=True)
        api.add_space_variable(space_id, "API_URL", api_url,
                               description="Public FraudLens scoring API")  # fmt: skip
        info = api.upload_folder(repo_id=space_id, folder_path=str(stage), repo_type="space",
                                 commit_message="FraudLens dashboard")  # fmt: skip
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return str(getattr(info, "commit_url", info))
