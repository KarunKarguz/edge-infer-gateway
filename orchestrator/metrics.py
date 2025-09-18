"""Prometheus metrics for orchestrator runtime."""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

PIPELINE_INGRESS = Counter(
    "eig_pipeline_ingress_total",
    "Number of messages entering each pipeline",
    labelnames=("pipeline",),
)

PIPELINE_DROPPED = Counter(
    "eig_pipeline_dropped_total",
    "Messages dropped due to deadline or errors",
    labelnames=("pipeline", "reason"),
)

PIPELINE_LATENCY = Histogram(
    "eig_pipeline_latency_ms",
    "End-to-end latency observed by pipeline agents (milliseconds)",
    labelnames=("pipeline",),
    buckets=(1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000),
)

QUEUE_DEPTH = Gauge(
    "eig_pipeline_queue_depth",
    "Messages waiting for pipeline processing",
)
