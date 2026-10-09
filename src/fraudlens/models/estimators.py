"""Model pipelines. Every model takes the same raw feature frame (see :func:`prepare_features`).

Only scikit-learn, XGBoost and LightGBM classes are used (no custom transformers), so models
can be saved with MLflow's default ``skops`` format, which loads only explicitly trusted types
instead of executing arbitrary pickled code.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier

from fraudlens.features.feature_names import CATEGORICAL_FEATURES, FEATURE_NAMES, NUMERIC_FEATURES

MODEL_NAMES = ("logreg", "xgboost", "lightgbm")
ESTIMATOR_STEP = "model"

# Types skops must be told to trust when saving/loading each pipeline (found with
# skops.io.get_untrusted_types). All are library classes; no project code is deserialised.
SKOPS_TRUSTED_TYPES: dict[str, list[str]] = {
    "logreg": ["numpy.dtype"],
    "xgboost": ["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"],
    "lightgbm": [
        "collections.OrderedDict",
        "lightgbm.basic.Booster",
        "lightgbm.sklearn.LGBMClassifier",
    ],
}


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select the model features in canonical order; categories as plain strings.

    The same function feeds training (from Parquet, where ``category`` is a pandas category)
    and serving (from JSON, where it is a string), so both see identical inputs.
    """
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        raise KeyError(f"missing feature columns: {missing}")
    X = df[FEATURE_NAMES].copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype(str)
    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="raise").astype("float64")
    return X


def single_row_frame(features: dict[str, Any]) -> pd.DataFrame:
    """Fast equivalent of ``prepare_features(pd.DataFrame([features]))`` for one transaction.

    Builds the frame column by column with final dtypes in one constructor call; on the API's
    hot path this is several times faster than the general function (tested to be identical).
    """
    data: dict[str, Any] = {}
    for name in FEATURE_NAMES:
        value = features[name]
        if name in CATEGORICAL_FEATURES:
            data[name] = pd.array([str(value)], dtype="str")
        else:
            data[name] = np.array([np.nan if value is None else value], dtype="float64")
    return pd.DataFrame(data, columns=FEATURE_NAMES)


def set_serving_threads(pipeline: Pipeline, n_jobs: int = 1) -> Pipeline:
    """Use ``n_jobs`` threads for prediction (1 is fastest for single rows: no thread start-up)."""
    model = pipeline.named_steps[ESTIMATOR_STEP]
    if "n_jobs" in model.get_params():
        model.set_params(n_jobs=n_jobs)
    return pipeline


def _tree_preprocessor() -> ColumnTransformer:
    # Categories -> integer codes; an unseen category becomes missing (NaN), which both tree
    # libraries route like any other missing value. Numeric features pass through untouched:
    # trees need no scaling and handle the NaN "no card history" values natively.
    return ColumnTransformer(
        [
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan),
                CATEGORICAL_FEATURES,
            ),
            ("num", "passthrough", NUMERIC_FEATURES),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def _linear_preprocessor() -> ColumnTransformer:
    # Logistic regression needs complete, scaled inputs: impute the no-history NaNs with the
    # median plus a "was missing" indicator, scale, and one-hot encode the category.
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", numeric, NUMERIC_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def build_pipeline(
    name: str, params: dict[str, Any], seed: int, scale_pos_weight: float
) -> Pipeline:
    """Create an unfitted pipeline for ``name`` (one of :data:`MODEL_NAMES`).

    Args:
        name: Model family.
        params: Hyperparameters from ``config.yaml`` (or tuned ones from Phase 5).
        seed: Random seed.
        scale_pos_weight: legit / fraud ratio in the training data, for class weighting.
    """
    if name == "logreg":
        model = LogisticRegression(class_weight="balanced", random_state=seed, **params)
        return Pipeline([("prep", _linear_preprocessor()), (ESTIMATOR_STEP, model)])
    if name == "xgboost":
        model = XGBClassifier(
            tree_method="hist",
            enable_categorical=True,
            # The ordinal-encoded first column is categorical ("c"); the rest numeric ("q").
            feature_types=["c"] * len(CATEGORICAL_FEATURES) + ["q"] * len(NUMERIC_FEATURES),
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            random_state=seed,
            n_jobs=-1,
            **params,
        )
        return Pipeline([("prep", _tree_preprocessor()), (ESTIMATOR_STEP, model)])
    if name == "lightgbm":
        model = LGBMClassifier(
            class_weight="balanced", random_state=seed, n_jobs=-1, verbose=-1, **params
        )
        return Pipeline([("prep", _tree_preprocessor()), (ESTIMATOR_STEP, model)])
    raise ValueError(f"unknown model {name!r}; expected one of {MODEL_NAMES}")


def fit_kwargs(name: str) -> dict[str, Any]:
    """Extra ``Pipeline.fit`` arguments (LightGBM is told which column is categorical)."""
    if name == "lightgbm":
        return {f"{ESTIMATOR_STEP}__categorical_feature": CATEGORICAL_FEATURES}
    return {}


def feature_importance(pipeline: Pipeline) -> pd.Series:
    """Importance per original feature, normalised to sum to 1.

    Trees: total gain. Logistic regression: absolute standardised coefficient; a categorical
    feature takes its largest one-hot coefficient (summing 14 columns would inflate it), and a
    missing-value indicator is added to its source feature.
    """
    model = pipeline.named_steps[ESTIMATOR_STEP]
    names = pipeline.named_steps["prep"].get_feature_names_out()
    if isinstance(model, XGBClassifier):
        gain = model.get_booster().get_score(importance_type="total_gain")
        values = np.array([gain.get(n, 0.0) for n in names])
    elif isinstance(model, LGBMClassifier):
        values = model.booster_.feature_importance(importance_type="gain").astype(float)
    else:
        values = np.abs(model.coef_.ravel())
    raw = pd.Series(values, index=names)

    def source(col: str) -> str:
        if col.startswith("missingindicator_"):
            return col.removeprefix("missingindicator_")
        for cat in CATEGORICAL_FEATURES:
            if col == cat or col.startswith(f"{cat}_"):
                return cat
        return col

    sources = raw.index.map(source)
    is_cat = sources.isin(CATEGORICAL_FEATURES)
    grouped = (
        pd.concat(
            [
                raw[is_cat].groupby(sources[is_cat]).max(),
                raw[~is_cat].groupby(sources[~is_cat]).sum(),
            ]
        )
        .groupby(level=0)
        .sum()
        .reindex(FEATURE_NAMES, fill_value=0.0)
    )
    total = grouped.sum()
    return (grouped / total if total else grouped).sort_values(ascending=False)
