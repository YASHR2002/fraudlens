"""FastAPI scoring service.

Endpoints: /health, /ready, /predict, /predict/batch, /predict_explained, /model-info, /metrics.
Built by :func:`create_app` so tests can inject a tiny model, an empty state and a fake LLM;
in production everything is loaded at startup from the configured sources.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from fraudlens import __version__
from fraudlens.api.metrics import LATENCY_BUCKETS, create_model_metrics
from fraudlens.api.model_loader import ModelBundle
from fraudlens.api.schemas import (
    BatchIn,
    BatchItemOut,
    BatchOut,
    ExplainedPredictionOut,
    ModelInfoOut,
    PredictionOut,
    ShapFactorOut,
    TransactionIn,
)
from fraudlens.api.state import LiveState
from fraudlens.explain.factors import CATEGORY_NAMES, build_factors, top_factors
from fraudlens.explain.llm_explainer import LLMExplainer
from fraudlens.explain.prompts import ExplanationContext
from fraudlens.features.feature_names import FEATURE_NAMES
from fraudlens.features.online import OutOfOrderError, Transaction
from fraudlens.models.estimators import prepare_features, single_row_frame

logger = logging.getLogger("fraudlens.api")


@dataclass
class Services:
    """What the endpoints need; populated at startup (or injected by tests)."""

    bundle: ModelBundle | None = None
    state: LiveState | None = None
    llm: LLMExplainer | None = None
    top_factors: int = 5
    load_error: str | None = None
    demo: pd.DataFrame | None = None  # demo transactions for the dashboard (optional)


def _to_txn(t: TransactionIn) -> Transaction:
    return Transaction(
        cc_num=t.cc_num, trans_ts=t.trans_ts.replace(tzinfo=None), amt=t.amt,
        category=t.category, dob=t.dob, city_pop=t.city_pop, lat=t.lat, long=t.long,
        merch_lat=t.merch_lat, merch_long=t.merch_long,
    )  # fmt: skip


def create_app(
    services: Services | None = None, loader: Callable[[], Services] | None = None
) -> FastAPI:
    """Build the app. Pass ready ``services`` (tests) or a ``loader`` run at startup."""
    svc = services or Services()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if loader is not None:
            try:
                loaded = loader()
                svc.bundle, svc.state, svc.llm = loaded.bundle, loaded.state, loaded.llm
                svc.top_factors = loaded.top_factors
                svc.demo = loaded.demo
                publish_model_gauges()
                logger.info("Ready: %s v%s (%s), threshold %.4f, %d cards", svc.bundle.name,
                            svc.bundle.version, svc.bundle.source, svc.bundle.threshold,
                            len(svc.state.store.cards))  # fmt: skip
            except Exception as exc:  # noqa: BLE001 - stay up, report not-ready on /ready
                svc.load_error = f"{type(exc).__name__}: {exc}"
                logger.exception("Startup failed; /ready will report 503")
        yield

    app = FastAPI(
        title="FraudLens scoring API",
        version=__version__,
        description="Real-time credit card fraud scoring with SHAP and LLM explanations. "
        "`/predict` is the fast path; `/predict_explained` adds SHAP factors and an analyst note.",
        lifespan=lifespan,
    )
    app.state.services = svc
    # Prometheus metrics (request counts, latency histograms) at /metrics; Phase 9 adds
    # model-level metrics (score distribution, flags, LLM fallbacks). Each app gets its own
    # registry, so several apps in one process (tests) never share or clash on metrics.
    from prometheus_client import CollectorRegistry
    from prometheus_fastapi_instrumentator import Instrumentator
    from prometheus_fastapi_instrumentator import metrics as http_metrics

    app.state.registry = CollectorRegistry()
    instrumentator = Instrumentator(excluded_handlers=["/metrics"], registry=app.state.registry)
    # Fine latency buckets: the default ones start at 100 ms, the fast path takes ~5 ms.
    instrumentator.add(
        http_metrics.default(latency_lowr_buckets=LATENCY_BUCKETS, registry=app.state.registry)
    )
    instrumentator.instrument(app).expose(app, include_in_schema=False)
    mm = create_model_metrics(app.state.registry)
    app.state.model_metrics = mm

    def publish_model_gauges() -> None:
        if svc.bundle is not None:
            b = svc.bundle
            mm.model_info.labels(b.name, b.version, b.family, b.source).set(1)
            mm.threshold.set(b.threshold)
        if svc.state is not None:
            mm.cards.set(len(svc.state.store.cards))

    publish_model_gauges()  # injected services (tests); the loader path publishes after loading

    @app.middleware("http")
    async def log_latency(request: Request, call_next: Callable) -> Any:
        start = time.perf_counter()
        response = await call_next(request)
        ms = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time-Ms"] = f"{ms:.2f}"
        if request.url.path not in ("/metrics", "/health"):
            logger.info("%s %s -> %d in %.1f ms", request.method, request.url.path,
                        response.status_code, ms)  # fmt: skip
        return response

    def ready() -> tuple[ModelBundle, LiveState]:
        if svc.bundle is None or svc.state is None:
            raise HTTPException(503, detail=svc.load_error or "model or state not loaded yet")
        return svc.bundle, svc.state

    def score_one(t: TransactionIn, endpoint: str) -> tuple[PredictionOut, dict[str, Any], float]:
        bundle, state = ready()
        start = time.perf_counter()
        try:
            scored = state.features(_to_txn(t), t.trans_num, record=t.update_state)
        except OutOfOrderError as exc:
            mm.rejected_out_of_order.inc()
            raise HTTPException(
                409, detail=f"transaction is older than this card's latest recorded one: {exc}"
            ) from exc
        X = single_row_frame(scored.features)
        score = float(bundle.pipeline.predict_proba(X)[0, 1])
        flagged = score >= bundle.threshold
        mm.predictions.labels(endpoint, "flagged" if flagged else "approved").inc()
        mm.scores.observe(score)
        if scored.duplicate:
            mm.duplicates.inc()
        if scored.recorded:
            mm.cards.set(len(state.store.cards))
        out = PredictionOut(
            trans_num=t.trans_num, score=score, flagged=flagged,
            decision="flagged" if flagged else "approved", threshold=bundle.threshold,
            model_name=bundle.name, model_version=bundle.version,
            state_updated=scored.recorded, duplicate=scored.duplicate,
            latency_ms=round((time.perf_counter() - start) * 1000, 3),
        )  # fmt: skip
        return out, scored.features, score

    @app.get("/health", tags=["ops"])
    def health() -> dict[str, str]:
        """Liveness: the process is up."""
        return {"status": "ok"}

    @app.get("/ready", tags=["ops"])
    def readiness() -> JSONResponse:
        """Readiness: model and card state are loaded."""
        if svc.bundle is None or svc.state is None:
            return JSONResponse({"status": "not ready", "reason": svc.load_error}, status_code=503)
        return JSONResponse({"status": "ready", "model_version": svc.bundle.version,
                             "cards": len(svc.state.store.cards)})  # fmt: skip

    @app.post("/predict", response_model=PredictionOut, tags=["scoring"])
    def predict(t: TransactionIn) -> PredictionOut:
        """Score one transaction (fast path, no explanation)."""
        return score_one(t, "predict")[0]

    @app.post("/predict/batch", response_model=BatchOut, tags=["scoring"])
    def predict_batch(batch: BatchIn) -> BatchOut:
        """Score up to 1,000 transactions; processed in time order, results in input order."""
        start = time.perf_counter()
        order = sorted(range(len(batch.transactions)),
                       key=lambda i: batch.transactions[i].trans_ts)  # fmt: skip
        results: dict[int, BatchItemOut] = {}
        for i in order:
            t = batch.transactions[i]
            try:
                results[i] = BatchItemOut(trans_num=t.trans_num, result=score_one(t, "batch")[0])
            except HTTPException as exc:
                results[i] = BatchItemOut(trans_num=t.trans_num, error=str(exc.detail))
        return BatchOut(results=[results[i] for i in range(len(order))],
                        latency_ms=round((time.perf_counter() - start) * 1000, 3))  # fmt: skip

    @app.post("/predict_explained", response_model=ExplainedPredictionOut, tags=["scoring"])
    def predict_explained(t: TransactionIn) -> ExplainedPredictionOut:
        """Score one transaction and explain it: SHAP factors plus an LLM analyst note."""
        start = time.perf_counter()
        pred, features, score = score_one(t, "predict_explained")
        bundle = svc.bundle
        X_raw = pd.DataFrame([features])[FEATURE_NAMES]
        shap_values = dict(zip(FEATURE_NAMES, map(float, bundle.explainer.shap_values(X_raw)[0]),
                               strict=True))  # fmt: skip
        row = prepare_features(X_raw).iloc[0].to_dict()
        factors = build_factors(row, shap_values)
        ctx = ExplanationContext(
            trans_num=t.trans_num, score=score, threshold=bundle.threshold,
            factors=top_factors(factors, svc.top_factors, exclude_protected=True),
            amount=t.amt, category=t.category, hour=int(row["hour"]),
            distance_km=float(row["distance_km"]),
        )  # fmt: skip
        llm_start = time.perf_counter()
        note = svc.llm.explain(ctx) if svc.llm else LLMExplainer(None, None).explain(ctx)
        mm.llm_explanations.labels(note.explanation_source).inc()
        mm.llm_latency.labels(note.explanation_source).observe(time.perf_counter() - llm_start)
        clean = {k: (None if isinstance(v, float) and v != v else v) for k, v in features.items()}
        return ExplainedPredictionOut(
            **{**pred.model_dump(), "latency_ms": round((time.perf_counter() - start) * 1000, 3)},
            base_value_log_odds=bundle.explainer.base_value,
            shap_factors=[ShapFactorOut(**f.as_dict()) for f in factors[:8]],
            features=clean,
            explanation=note,
        )

    @app.get("/demo/transactions", tags=["demo"])
    def demo_transactions(
        label: Literal["all", "fraud", "legit"] = "all", limit: int = 200
    ) -> list[dict[str, Any]]:
        """Demo transactions for the dashboard: each card's first transaction after the state
        snapshot, so their features are exact. Includes the true label for display only."""
        if svc.demo is None:
            raise HTTPException(404, detail="no demo transactions loaded")
        df = svc.demo
        if label != "all":
            df = df[df["is_fraud"] == (1 if label == "fraud" else 0)]
        df = df.head(max(1, min(limit, 1000)))
        out = df.assign(trans_ts=df["trans_ts"].map(lambda v: v.isoformat()),
                        dob=df["dob"].map(lambda v: v.isoformat()))  # fmt: skip
        return out.to_dict("records")

    @app.get("/model-info", response_model=ModelInfoOut, tags=["model"])
    def model_info() -> ModelInfoOut:
        """Which model is serving, its threshold, metrics, global importance and fairness."""
        bundle, state = ready()
        llm = svc.llm
        return ModelInfoOut(
            model_name=bundle.name, model_version=bundle.version, alias=bundle.alias,
            model_family=bundle.family, source=bundle.source, run_id=bundle.run_id,
            trained_utc=bundle.trained_utc, threshold=bundle.threshold,
            threshold_source=bundle.threshold_source, metrics=bundle.metrics,
            global_importance=bundle.global_importance, fairness=bundle.fairness,
            state=state.summary(),
            llm={"enabled": bool(llm and llm.enabled), "model": llm.model if llm else None},
            categories=CATEGORY_NAMES,
        )  # fmt: skip

    return app
