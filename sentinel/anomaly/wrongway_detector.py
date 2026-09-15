"""Wrong-way movement: a track's heading deviates from the configured expected flow direction
by more than `angle_threshold_deg`, sustained for `min_persist_sec`.
"""
from __future__ import annotations

from sentinel.anomaly.events import AnomalyEvent
from sentinel.tracking.trajectory_store import Track


def _angle_diff(a: float, b: float) -> float:
    d = abs(a - b) % 360
    return min(d, 360 - d)


class WrongWayDetector:
    def __init__(
        self,
        angle_threshold_deg: float = 90,
        min_speed_px_per_sec: float = 15,
        min_persist_sec: float = 2.0,
    ) -> None:
        self.angle_threshold_deg = angle_threshold_deg
        self.min_speed_px_per_sec = min_speed_px_per_sec
        self.min_persist_sec = min_persist_sec

    def evaluate(
        self,
        track: Track,
        source: str,
        expected_flow_deg: float,
        now: float | None = None,
    ) -> AnomalyEvent | None:
        if len(track.points) < 2:
            return None
        now = track.points[-1].timestamp if now is None else now

        speed = track.speed(window_sec=0.5)
        heading = track.heading_deg(window_sec=0.5)
        if speed < self.min_speed_px_per_sec or heading is None:
            track.state.pop("wrongway_since", None)
            track.state["wrongway_alerted"] = False
            return None

        deviation = _angle_diff(heading, expected_flow_deg)
        if deviation < self.angle_threshold_deg:
            track.state.pop("wrongway_since", None)
            track.state["wrongway_alerted"] = False
            return None

        since = track.state.get("wrongway_since")
        if since is None:
            since = now
            track.state["wrongway_since"] = since

        persisted = now - since
        if persisted < self.min_persist_sec or track.state.get("wrongway_alerted"):
            return None

        track.state["wrongway_alerted"] = True
        latest = track.points[-1]
        confidence = min(1.0, 0.5 + deviation / 360)
        return AnomalyEvent(
            event_type="wrong_way",
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
                "heading_deg": round(heading, 1),
                "expected_flow_deg": expected_flow_deg,
                "deviation_deg": round(deviation, 1),
            },
        )
