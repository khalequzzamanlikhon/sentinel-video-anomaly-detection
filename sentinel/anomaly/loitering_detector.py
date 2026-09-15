"""Loitering: a track that stays within a small radius for longer than `min_duration_sec`."""
from __future__ import annotations

from sentinel.anomaly.events import AnomalyEvent
from sentinel.tracking.trajectory_store import Track


class LoiteringDetector:
    def __init__(self, radius_px: float = 60, min_duration_sec: float = 30) -> None:
        self.radius_px = radius_px
        self.min_duration_sec = min_duration_sec

    def evaluate(self, track: Track, source: str, now: float | None = None) -> AnomalyEvent | None:
        if track.cls_name != "person" or len(track.points) < 2:
            return None
        now = track.points[-1].timestamp if now is None else now

        radius = track.max_radius_over(self.min_duration_sec, now=now)
        duration = track.duration_in_window(self.min_duration_sec, now=now)

        if radius > self.radius_px or duration < self.min_duration_sec:
            track.state["loitering_alerted"] = False
            return None
        if track.state.get("loitering_alerted"):
            return None

        track.state["loitering_alerted"] = True
        latest = track.points[-1]
        confidence = min(1.0, 0.6 + (self.radius_px - radius) / max(self.radius_px, 1) * 0.4)
        return AnomalyEvent(
            event_type="loitering",
            track_id=track.track_id,
            source=source,
            confidence=confidence,
            timestamp=now,
            bbox=(
                latest.x - latest.w / 2,
                latest.y - latest.h / 2,
                latest.x + latest.w / 2,
                latest.y + latest.h / 2,
            ),
            details={"radius_px": round(radius, 1), "duration_sec": round(duration, 1)},
        )
