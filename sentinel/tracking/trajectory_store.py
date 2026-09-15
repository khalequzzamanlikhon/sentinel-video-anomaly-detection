"""Per-track trajectory history. Every anomaly rule reads from this -- it's the shared state
between "tracking" and "temporal reasoning" in the architecture diagram.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class TrackPoint:
    x: float  # center x, px
    y: float  # center y, px
    w: float  # bbox width, px
    h: float  # bbox height, px
    timestamp: float
    confidence: float = 0.0

    @property
    def aspect_ratio(self) -> float:
        return self.h / self.w if self.w > 1e-6 else 0.0


@dataclass
class Track:
    track_id: int
    cls: int
    cls_name: str
    points: deque[TrackPoint] = field(default_factory=lambda: deque(maxlen=150))
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    # per-track scratch state used by stateful rules (loitering anchor, abandoned-object anchor, etc.)
    state: dict = field(default_factory=dict)

    def add(self, point: TrackPoint) -> None:
        self.points.append(point)
        self.last_seen = point.timestamp

    def age_sec(self, now: float | None = None) -> float:
        return (now or time.time()) - self.first_seen

    def idle_sec(self, now: float | None = None) -> float:
        return (now or time.time()) - self.last_seen

    def latest(self) -> TrackPoint | None:
        return self.points[-1] if self.points else None

    def velocity(self, window_sec: float = 0.5) -> tuple[float, float]:
        """Average (vx, vy) in px/sec over the last `window_sec` of history."""
        if len(self.points) < 2:
            return 0.0, 0.0
        latest = self.points[-1]
        cutoff = latest.timestamp - window_sec
        ref = self.points[0]
        for p in reversed(self.points):
            if p.timestamp <= cutoff:
                ref = p
                break
        dt = latest.timestamp - ref.timestamp
        if dt <= 1e-6:
            return 0.0, 0.0
        return (latest.x - ref.x) / dt, (latest.y - ref.y) / dt

    def speed(self, window_sec: float = 0.5) -> float:
        vx, vy = self.velocity(window_sec)
        return math.hypot(vx, vy)

    def heading_deg(self, window_sec: float = 0.5) -> float | None:
        vx, vy = self.velocity(window_sec)
        if math.hypot(vx, vy) < 1e-6:
            return None
        return math.degrees(math.atan2(vy, vx)) % 360

    def displacement_since(self, seconds_ago: float, now: float | None = None) -> float:
        """Straight-line distance moved from the point closest to `now - seconds_ago` to now."""
        if len(self.points) < 2:
            return 0.0
        now = self.points[-1].timestamp if now is None else now
        cutoff = now - seconds_ago
        ref = self.points[0]
        for p in self.points:
            if p.timestamp >= cutoff:
                ref = p
                break
        latest = self.points[-1]
        return math.hypot(latest.x - ref.x, latest.y - ref.y)

    def max_radius_over(self, seconds: float, now: float | None = None) -> float:
        """Max distance from the earliest point within the last `seconds` to any later point
        in that window -- used by loitering (small radius = staying put)."""
        if now is None:
            now = self.points[-1].timestamp if self.points else time.time()
        cutoff = now - seconds
        window = [p for p in self.points if p.timestamp >= cutoff]
        if len(window) < 2:
            return 0.0
        anchor = window[0]
        return max(math.hypot(p.x - anchor.x, p.y - anchor.y) for p in window)

    def duration_in_window(self, seconds: float, now: float | None = None) -> float:
        if now is None:
            now = self.points[-1].timestamp if self.points else time.time()
        cutoff = now - seconds
        window = [p for p in self.points if p.timestamp >= cutoff]
        if len(window) < 2:
            return 0.0
        return window[-1].timestamp - window[0].timestamp


class TrajectoryStore:
    def __init__(self, history_len: int = 150, stale_ttl_sec: float = 10.0) -> None:
        self.history_len = history_len
        self.stale_ttl_sec = stale_ttl_sec
        self.tracks: dict[int, Track] = {}

    def update(
        self,
        track_id: int,
        cls: int,
        cls_name: str,
        xyxy: tuple[float, float, float, float],
        confidence: float,
        timestamp: float | None = None,
    ) -> Track:
        timestamp = time.time() if timestamp is None else timestamp
        x1, y1, x2, y2 = xyxy
        point = TrackPoint(
            x=(x1 + x2) / 2.0,
            y=(y1 + y2) / 2.0,
            w=max(x2 - x1, 1e-6),
            h=max(y2 - y1, 1e-6),
            timestamp=timestamp,
            confidence=confidence,
        )
        track = self.tracks.get(track_id)
        if track is None:
            track = Track(
                track_id=track_id,
                cls=cls,
                cls_name=cls_name,
                points=deque(maxlen=self.history_len),
            )
            self.tracks[track_id] = track
        track.add(point)
        return track

    def prune_stale(self, now: float | None = None) -> list[int]:
        now = time.time() if now is None else now
        stale = [tid for tid, t in self.tracks.items() if t.idle_sec(now) > self.stale_ttl_sec]
        for tid in stale:
            del self.tracks[tid]
        return stale

    def get(self, track_id: int) -> Track | None:
        return self.tracks.get(track_id)

    def by_class(self, cls_names: list[str]) -> list[Track]:
        return [t for t in self.tracks.values() if t.cls_name in cls_names]

    def active_ids(self) -> list[int]:
        return list(self.tracks.keys())
