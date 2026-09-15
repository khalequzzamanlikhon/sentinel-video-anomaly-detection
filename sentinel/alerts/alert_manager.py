"""Turns raw AnomalyEvents into stored, deduplicated, severity-scored alerts with thumbnails
and (after a short delay, so post-event frames exist) video clips.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

from sentinel.alerts import video_clipper
from sentinel.alerts.deduplicator import Deduplicator
from sentinel.alerts.storage import AlertStorage
from sentinel.anomaly.events import AnomalyEvent
from sentinel.capture.frame_buffer import FrameBuffer
from sentinel.core.metrics import ALERT_COUNT

logger = logging.getLogger(__name__)

DEFAULT_SEVERITY = {
    "fall": "critical",
    "abandoned_object": "high",
    "crowd_density": "medium",
    "wrong_way": "medium",
    "loitering": "low",
}


class AlertManager:
    def __init__(
        self,
        alerts_config: dict,
        frame_buffers: dict[str, FrameBuffer],
        storage: AlertStorage | None = None,
    ) -> None:
        alerting_cfg = alerts_config.get("alerting", {})
        self.severity_map = {**DEFAULT_SEVERITY, **alerting_cfg.get("severity", {})}
        self.dedup = Deduplicator(dedup_window_sec=alerting_cfg.get("dedup_window_sec", 30))

        clip_cfg = alerting_cfg.get("clip", {})
        self.pre_event_sec = clip_cfg.get("pre_event_sec", 5)
        self.post_event_sec = clip_cfg.get("post_event_sec", 5)

        storage_cfg = alerting_cfg.get("storage", {})
        self.storage = storage or AlertStorage(storage_cfg.get("db_path", "data/alerts.db"))
        self.clips_dir = storage_cfg.get("clips_dir", "data/clips")
        self.thumbnails_dir = storage_cfg.get("thumbnails_dir", "data/thumbnails")

        self.frame_buffers = frame_buffers
        self._timers: list[threading.Timer] = []

    def process(self, events: list[AnomalyEvent]) -> list[int]:
        """Dedup + score + persist a batch of events. Returns the inserted alert ids."""
        alert_ids = []
        for event in self.dedup.filter(events):
            alert_ids.append(self._handle_one(event))
        return alert_ids

    def _handle_one(self, event: AnomalyEvent) -> int:
        severity = self.severity_map.get(event.event_type, "low")
        buffer = self.frame_buffers.get(event.source)
        thumbnail_path = None
        if buffer is not None:
            thumbnail_path = video_clipper.save_thumbnail(buffer, event, self.thumbnails_dir)

        alert_id = self.storage.insert(
            source=event.source,
            event_type=event.event_type,
            track_id=event.track_id,
            severity=severity,
            confidence=event.confidence,
            timestamp=event.timestamp,
            thumbnail_path=thumbnail_path,
            details=event.details,
        )
        ALERT_COUNT.labels(source=event.source, event_type=event.event_type, severity=severity).inc()
        logger.info(
            "ALERT [%s] %s on %s track=%s confidence=%.2f",
            severity.upper(), event.event_type, event.source, event.track_id, event.confidence,
        )

        if buffer is not None:
            timer = threading.Timer(
                self.post_event_sec + 0.5, self._extract_clip_later, args=(alert_id, event, buffer)
            )
            timer.daemon = True
            timer.start()
            self._timers.append(timer)

        return alert_id

    def _extract_clip_later(self, alert_id: int, event: AnomalyEvent, buffer: FrameBuffer) -> None:
        try:
            clip_path = video_clipper.extract_clip(
                buffer, event, self.clips_dir, self.pre_event_sec, self.post_event_sec
            )
            if clip_path:
                self.storage.update_clip_path(alert_id, clip_path)
        except Exception:
            logger.exception("Failed to extract clip for alert %d", alert_id)

    def shutdown(self) -> None:
        for t in self._timers:
            t.cancel()
        self.storage.close()
