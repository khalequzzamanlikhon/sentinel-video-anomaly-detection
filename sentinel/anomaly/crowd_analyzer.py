"""Crowd density: person count inside an ROI exceeding a threshold, sustained over time."""
from __future__ import annotations

import time

from sentinel.anomaly.events import AnomalyEvent
from sentinel.tracking.trajectory_store import TrajectoryStore


def _in_roi(x: float, y: float, roi_px: tuple[float, float, float, float]) -> bool:
    x1, y1, x2, y2 = roi_px
    return x1 <= x <= x2 and y1 <= y <= y2


class CrowdAnalyzer:
    def __init__(self, person_count_threshold: int = 8, sustained_sec: float = 30) -> None:
        self.person_count_threshold = person_count_threshold
        self.sustained_sec = sustained_sec
        self._streak_start: dict[str, float] = {}
        self._alerted: dict[str, bool] = {}

    def evaluate(
        self,
        store: TrajectoryStore,
        source: str,
        roi_norm: list[float] | None = None,
        frame_shape: tuple[int, int] | None = None,
        now: float | None = None,
    ) -> AnomalyEvent | None:
        now = time.time() if now is None else now
        persons = store.by_class(["person"])

        roi_px = None
        if roi_norm and frame_shape:
            h, w = frame_shape
            roi_px = (roi_norm[0] * w, roi_norm[1] * h, roi_norm[2] * w, roi_norm[3] * h)

        if roi_px:
            count = sum(1 for p in persons if p.points and _in_roi(p.points[-1].x, p.points[-1].y, roi_px))
        else:
            count = len(persons)

        if count < self.person_count_threshold:
            self._streak_start.pop(source, None)
            self._alerted[source] = False
            return None

        streak_start = self._streak_start.setdefault(source, now)
        duration = now - streak_start
        if duration < self.sustained_sec or self._alerted.get(source):
            return None

        self._alerted[source] = True
        confidence = min(1.0, 0.5 + (count - self.person_count_threshold) * 0.05)
        return AnomalyEvent(
            event_type="crowd_density",
            track_id=None,
            source=source,
            confidence=confidence,
            timestamp=now,
            details={"person_count": count, "sustained_sec": round(duration, 1)},
        )
