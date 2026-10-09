"""Hyperparameter tuning with Optuna (run on Kaggle; see notebooks/kaggle_tuning.ipynb).

Each trial builds the *same* pipeline as local training (:func:`build_pipeline`), fits it on
the train split, and scores validation PR-AUC. Unpromising trials are stopped early (pruned)
from the validation metric reported every few boosting rounds. The test split is never
available to this code: the Kaggle bundle contains only train and validation rows.
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score
from xgboost.callback import TrainingCallback

from fraudlens.features.feature_names import CATEGORICAL_FEATURES
from fraudlens.models.estimators import ESTIMATOR_STEP, build_pipeline

logger = logging.getLogger(__name__)

TUNABLE_MODELS = ("xgboost", "lightgbm")
META_KEY = "_meta"


# ------------------------------------------------------------------------- search spaces


def suggest_xgboost(trial: optuna.Trial) -> dict[str, Any]:
    """XGBoost search space (class weighting stays fixed at the train legit/fraud ratio)."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    }


def suggest_lightgbm(trial: optuna.Trial) -> dict[str, Any]:
    """LightGBM search space; leaf count and minimum leaf size control over-fitting."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 500, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    }


SEARCH_SPACES: dict[str, Callable[[optuna.Trial], dict[str, Any]]] = {
    "xgboost": suggest_xgboost,
    "lightgbm": suggest_lightgbm,
}


# ------------------------------------------------------------------------------ pruning


class _XGBoostPruning(TrainingCallback):
    """Report validation PR-AUC (``aucpr``) every ``period`` rounds; stop pruned trials."""

    def __init__(self, trial: optuna.Trial, period: int) -> None:
        super().__init__()
        self.trial, self.period = trial, period

    def after_iteration(self, model: Any, epoch: int, evals_log: dict) -> bool:
        if (epoch + 1) % self.period == 0:
            self.trial.report(float(evals_log["validation_0"]["aucpr"][-1]), step=epoch + 1)
            if self.trial.should_prune():
                raise optuna.TrialPruned(f"pruned at round {epoch + 1}")
        return False


def _lightgbm_pruning(trial: optuna.Trial, period: int) -> Callable[[Any], None]:
    """LightGBM callback: report validation average precision every ``period`` rounds."""

    def callback(env: Any) -> None:
        step = env.iteration + 1
        if step % period:
            return
        for _, metric, value, _ in env.evaluation_result_list:
            if metric == "average_precision":
                trial.report(float(value), step=step)
                if trial.should_prune():
                    raise optuna.TrialPruned(f"pruned at round {step}")

    return callback


# ---------------------------------------------------------------------------- objective


def make_objective(
    name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    seed: int,
    prune_period: int = 25,
) -> Callable[[optuna.Trial], float]:
    """Optuna objective: validation PR-AUC of the full pipeline built from the trial's params."""
    if name not in SEARCH_SPACES:
        raise ValueError(f"cannot tune {name!r}; expected one of {TUNABLE_MODELS}")
    scale_pos_weight = float((y_train == 0).sum() / y_train.sum())

    def objective(trial: optuna.Trial) -> float:
        params = SEARCH_SPACES[name](trial)
        pipeline = build_pipeline(name, params, seed, scale_pos_weight)
        # Fit the preprocessing once so the booster can be given a validation set to report on.
        prep = pipeline.named_steps["prep"]
        Xt, Xv = prep.fit_transform(X_train), prep.transform(X_val)
        model = pipeline.named_steps[ESTIMATOR_STEP]
        start = time.perf_counter()
        if name == "xgboost":
            model.set_params(callbacks=[_XGBoostPruning(trial, prune_period)])
            model.fit(Xt, y_train, eval_set=[(Xv, y_val)], verbose=False)
        else:
            model.fit(
                Xt,
                y_train,
                eval_X=Xv,  # LightGBM >= 4.6; `eval_set` is deprecated
                eval_y=y_val,
                eval_metric="average_precision",
                categorical_feature=CATEGORICAL_FEATURES,
                callbacks=[_lightgbm_pruning(trial, prune_period)],
            )
        trial.set_user_attr("fit_seconds", round(time.perf_counter() - start, 1))
        score = float(average_precision_score(y_val, model.predict_proba(Xv)[:, 1]))
        trial.set_user_attr("val_pr_auc", score)
        return score

    return objective


def tune(
    name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    n_trials: int = 50,
    timeout_seconds: float | None = None,
    seed: int = 42,
    prune_period: int = 25,
) -> optuna.Study:
    """Run an Optuna study maximising validation PR-AUC for ``name``.

    TPE sampling is seeded, so the sequence of suggested parameters is reproducible. The
    median pruner stops a trial whose PR-AUC at a given round is below the median of earlier
    trials at that round (after 5 complete trials and the first 100 rounds).
    """
    study = optuna.create_study(
        study_name=f"fraudlens-{name}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=100),
    )
    objective = make_objective(name, X_train, y_train, X_val, y_val, seed, prune_period)
    study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds, gc_after_trial=True)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    logger.info(
        "%s: %d trials (%d complete, %d pruned), best validation PR-AUC %.4f",
        name, len(study.trials), len(done), len(pruned), study.best_value,
    )  # fmt: skip
    return study


def best_params_payload(studies: dict[str, optuna.Study], meta: dict[str, Any]) -> dict[str, Any]:
    """The ``configs/best_params.json`` content: best params per model plus provenance."""
    payload: dict[str, Any] = {}
    summary: dict[str, Any] = {}
    for name, study in studies.items():
        params = dict(study.best_params)
        if name == "lightgbm":
            params["subsample_freq"] = 1  # fixed in the search space, not suggested
        payload[name] = params
        states = pd.Series([t.state.name for t in study.trials]).value_counts().to_dict()
        summary[name] = {
            "best_val_pr_auc": round(float(study.best_value), 6),
            "best_trial": study.best_trial.number,
            "trials": states,
        }
    payload[META_KEY] = {
        "created_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "objective": "validation PR-AUC (average precision)",
        "studies": summary,
        **meta,
    }
    return payload


def params_from_payload(payload: dict[str, Any], name: str) -> dict[str, Any]:
    """Hyperparameters for ``name`` from a best_params payload (raises a clear KeyError)."""
    if name not in payload:
        available = sorted(k for k in payload if k != META_KEY)
        raise KeyError(f"no tuned parameters for {name!r}; available: {available}")
    return dict(payload[name])
