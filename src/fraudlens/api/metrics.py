"""Model-level Prometheus metrics for the scoring API (scraped at /metrics).

HTTP request counts and latency come from ``prometheus-fastapi-instrumentator``; these add what
a fraud team watches: how many transactions are scored and flagged, how the score distribution
moves (an early sign of drift), and how often the LLM explanation falls back.

Each app has its own registry (see ``create_app``), so tests can build many apps safely.
"""

from __future__ import annotations

from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

# HTTP latency buckets in seconds: the fast path is ~5 ms, the explained path ~2 s.
LATENCY_BUCKETS = (0.001, 0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.025,
                   0.035, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)  # fmt: skip
# Fraud scores: dense near 0 (most traffic) and around typical thresholds.
SCORE_BUCKETS = (0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0)
LLM_BUCKETS = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0)


@dataclass
class ModelMetrics:
    predictions: Counter
    scores: Histogram
    llm_explanations: Counter
    llm_latency: Histogram
    duplicates: Counter
    rejected_out_of_order: Counter
    model_info: Gauge
    threshold: Gauge
    cards: Gauge


def create_model_metrics(registry: CollectorRegistry) -> ModelMetrics:
    """Register the FraudLens metrics on ``registry``."""
    return ModelMetrics(
        predictions=Counter(
            "fraudlens_predictions_total", "Transactions scored, by endpoint and decision.",
            ["endpoint", "decision"], registry=registry,
        ),
        scores=Histogram(
            "fraudlens_fraud_score", "Distribution of fraud scores returned.",
            buckets=SCORE_BUCKETS, registry=registry,
        ),
        llm_explanations=Counter(
            "fraudlens_llm_explanations_total",
            "Analyst notes produced, by source (llm or fallback).", ["source"], registry=registry,
        ),
        llm_latency=Histogram(
            "fraudlens_llm_explanation_seconds", "Time to produce an analyst note.",
            ["source"], buckets=LLM_BUCKETS, registry=registry,
        ),
        duplicates=Counter(
            "fraudlens_duplicate_transactions_total",
            "Transactions already recorded (client retries); scored but not re-recorded.",
            registry=registry,
        ),
        rejected_out_of_order=Counter(
            "fraudlens_out_of_order_total",
            "Transactions rejected (409) because they are older than the card's latest one.",
            registry=registry,
        ),
        model_info=Gauge(
            "fraudlens_model_info", "Serving model (value is always 1).",
            ["name", "version", "family", "source"], registry=registry,
        ),
        threshold=Gauge("fraudlens_decision_threshold", "Current decision threshold.",
                        registry=registry),
        cards=Gauge("fraudlens_cards_in_state", "Cards held in the in-memory state.",
                    registry=registry),
    )  # fmt: skip
