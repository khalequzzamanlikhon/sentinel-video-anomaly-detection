"""Fall detection: sudden bounding-box aspect-ratio collapse (a standing person's box is tall
and narrow; a fallen person's box is short and wide) that then *persists* with near-zero
velocity -- this is the multi-signal fusion + temporal consistency called for in the spec's
Challenge 2 (a person tying their shoe drops aspect ratio briefly but keeps moving; a person
squatting recovers before the persistence window elapses).
"""
from __future__ import annotations

from sentinel.anomaly.events import AnomalyEvent
from sentinel.tracking.trajectory_store import Track

BASELINE_WINDOW_SEC = 3.0


class FallDetector:
    def __init__(
        self,
        aspect_ratio_drop_pct: float = 50,
        window_sec: float = 0.5,
        min_persist_sec: float = 0.5,
        velocity_threshold_px_per_sec: float = 40,
    ) -> None:
        self.aspect_ratio_drop_pct = aspect_ratio_drop_pct
        self.window_sec = window_sec
        self.min_persist_sec = min_persist_sec
        self.velocity_threshold_px_per_sec = velocity_threshold_px_per_sec

    def evaluate(self, track: Track, source: str, now: float | None = None) -> AnomalyEvent | None:
        if track.cls_name != "person" or len(track.points) < 2:
            return None
        latest = track.points[-1]
        now = latest.timestamp if now is None else now

        baseline_candidates = [
            p.aspect_ratio
            for p in track.points
            if now - BASELINE_WINDOW_SEC <= p.timestamp <= now - self.window_sec
        ]
        if not baseline_candidates:
            return None
        baseline = max(baseline_candidates)
        if baseline <= 1e-6:
            return None

        drop_pct = (baseline - latest.aspect_ratio) / baseline * 100

        if drop_pct < self.aspect_ratio_drop_pct:
            track.state.pop("fall_candidate_since", None)
            track.state["fall_alerted"] = False
            return None

        candidate_since = track.state.get("fall_candidate_since")
        if candidate_since is None:
            candidate_since = now
            track.state["fall_candidate_since"] = candidate_since

        persisted_sec = now - candidate_since
        speed = track.speed(window_sec=self.min_persist_sec)

        if persisted_sec < self.min_persist_sec or speed > self.velocity_threshold_px_per_sec:
            return None
        if track.state.get("fall_alerted"):
            return None

        track.state["fall_alerted"] = True
        confidence = min(1.0, 0.5 + drop_pct / 200)
        return AnomalyEvent(
            event_type="fall",
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
            details={
                "aspect_ratio_drop_pct": round(drop_pct, 1),
                "persisted_sec": round(persisted_sec, 2),
                "speed_px_per_sec": round(speed, 1),
            },
        )
