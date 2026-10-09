"""SHAP explanations for the tree-model pipelines (TreeSHAP, exact).

Values are in **log-odds** of the model's raw score, so they add up exactly:
``base_value + sum(shap values) = raw score``, and ``sigmoid(raw score) = fraud score``.
Because training up-weights fraud (class weighting), the base value reflects that weighted
model rather than the 0.4% fraud rate; contributions are what matter for explanations.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from fraudlens.explain.factors import Factor, build_factors
from fraudlens.features.feature_names import FEATURE_NAMES, label
from fraudlens.models.estimators import ESTIMATOR_STEP, prepare_features

if TYPE_CHECKING:
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LocalExplanation:
    """SHAP explanation of one transaction."""

    score: float
    base_value: float
    shap_by_feature: dict[str, float]
    factors: list[Factor]  # grouped, largest first


class ShapExplainer:
    """Wraps a fitted tree pipeline (``prep`` + XGBoost/LightGBM) with ``shap.TreeExplainer``."""

    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline
        self.prep = pipeline.named_steps["prep"]
        self.model = pipeline.named_steps[ESTIMATOR_STEP]
        self.explainer = shap.TreeExplainer(self.model)
        # The preprocessor outputs columns in its own order (category first); remember how to
        # map them back to the canonical FEATURE_NAMES order.
        prep_names = list(self.prep.get_feature_names_out())
        self._order = [prep_names.index(f) for f in FEATURE_NAMES]

    @property
    def base_value(self) -> float:
        ev = self.explainer.expected_value
        return float(np.ravel(ev)[-1])

    def shap_values(self, X_raw: pd.DataFrame) -> np.ndarray:
        """SHAP values, shape (rows, 18), columns in FEATURE_NAMES order."""
        with warnings.catch_warnings():
            # shap warns that LightGBM's output format changed; both formats are handled below.
            warnings.filterwarnings("ignore", message=".*output has changed to a list.*")
            values = self.explainer.shap_values(self.prep.transform(prepare_features(X_raw)))
        if isinstance(values, list):  # some versions return one array per class
            values = values[-1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[..., -1]
        return values[:, self._order]

    def explain(self, row_raw: pd.DataFrame) -> LocalExplanation:
        """Explain a single transaction (a one-row DataFrame with the 18 features)."""
        if len(row_raw) != 1:
            raise ValueError("explain() takes exactly one row")
        values = self.shap_values(row_raw)[0]
        shap_by_feature = dict(zip(FEATURE_NAMES, map(float, values), strict=True))
        row = prepare_features(row_raw).iloc[0].to_dict()
        score = float(self.pipeline.predict_proba(prepare_features(row_raw))[0, 1])
        return LocalExplanation(
            score=score,
            base_value=self.base_value,
            shap_by_feature=shap_by_feature,
            factors=build_factors(row, shap_by_feature),
        )

    def explanation_object(self, X_raw: pd.DataFrame) -> shap.Explanation:
        """A ``shap.Explanation`` for plotting (category shown as its integer code)."""
        values = self.shap_values(X_raw)
        encoded = self.prep.transform(prepare_features(X_raw))
        data = encoded.to_numpy()[:, self._order] if hasattr(encoded, "to_numpy") else encoded
        return shap.Explanation(
            values=values,
            base_values=np.full(len(values), self.base_value),
            data=np.asarray(data, dtype=float),
            feature_names=[label(f) for f in FEATURE_NAMES],
        )


def global_plots(explainer: ShapExplainer, X_raw: pd.DataFrame, title: str) -> dict[str, Figure]:
    """Beeswarm (direction and spread) and bar (mean |SHAP|) summary plots."""
    # Plotting libraries are imported here, not at module level: the API uses this module for
    # per-transaction SHAP values only, and its image does not ship matplotlib.
    import matplotlib.pyplot as plt

    from fraudlens.viz import BLUE, INK2, apply_style

    apply_style()
    exp = explainer.explanation_object(X_raw)
    figures: dict[str, Figure] = {}
    for name, draw, subtitle in [
        ("shap_beeswarm.png", shap.plots.beeswarm, "each dot is one transaction"),
        ("shap_bar.png", shap.plots.bar, "mean absolute SHAP value"),
    ]:
        plt.figure(figsize=(8, 6.5))
        draw(exp, max_display=18, show=False)
        fig = plt.gcf()
        if name == "shap_bar.png":  # shap's default red -> project palette; text stays neutral
            for ax in fig.axes:
                for patch in ax.patches:
                    patch.set_color(BLUE)
                for text in ax.texts:
                    text.set_color(INK2)
        fig.suptitle(f"{title}: {subtitle}", x=0.01, ha="left", fontweight="bold", fontsize=11)
        fig.tight_layout()
        figures[name] = fig
        plt.close(fig)  # shap draws with pyplot; detach so figures don't accumulate
    return figures


def mean_abs_shap(explainer: ShapExplainer, X_raw: pd.DataFrame) -> pd.Series:
    """Global importance: mean |SHAP| per feature, largest first."""
    values = explainer.shap_values(X_raw)
    return pd.Series(np.abs(values).mean(axis=0), index=FEATURE_NAMES).sort_values(ascending=False)


def to_api_dict(exp: LocalExplanation, top_n: int = 5) -> dict[str, Any]:
    """JSON-friendly summary of a local explanation."""
    return {
        "score": exp.score,
        "base_value_log_odds": exp.base_value,
        "top_factors": [f.as_dict() for f in exp.factors[:top_n]],
    }
