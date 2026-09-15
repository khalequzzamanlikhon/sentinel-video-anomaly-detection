"""Prometheus metrics for the pipeline. Real, self-contained -- no external server required
to collect metrics; `start_metrics_server` exposes them over HTTP for Prometheus to scrape.
"""
from __future__ import annotations

import logging

from prometheus_client import Counter, Gauge, start_http_server

logger = logging.getLogger(__name__)

INFERENCE_FPS = Gauge(
    "inference_fps", "Effective inference frames-per-second per source", ["source"]
)
ALERT_COUNT = Counter(
    "alert_count", "Total alerts generated", ["source", "event_type", "severity"]
)
STREAM_LAG_SECONDS = Gauge(
    "stream_lag_seconds", "Delay between frame capture and processing", ["source"]
)
FRAMES_PROCESSED = Counter("frames_processed_total", "Total frames processed", ["source"])
FRAMES_DROPPED = Counter(
    "frames_dropped_total", "Frames dropped due to a full queue", ["source"]
)

_server_started = False


def start_metrics_server(port: int = 8000) -> None:
    global _server_started
    if _server_started:
        return
    try:
        start_http_server(port)
        _server_started = True
        logger.info("Prometheus metrics available at http://localhost:%d/metrics", port)
    except OSError as e:
        logger.warning("Could not start metrics server on port %d: %s", port, e)
