"""Abandoned object: a bag/suitcase/backpack track that has been stationary for a long time
while no person track remains near it (the "owner" walked away).
"""
from __future__ import annotations

import math

from sentinel.anomaly.events import AnomalyEvent
from sentinel.tracking.trajectory_store import Track, TrajectoryStore


class AbandonedObjectDetector:
    def __init__(
        self,
        classes: list[int] | None = None,
        stationary_sec: float = 300,
        owner_distance_px: float = 400,
        stationary_radius_px: float = 20,
    ) -> None:
        self.classes = classes or [24, 26, 28]  # backpack, handbag, suitcase
        self.stationary_sec = stationary_sec
        self.owner_distance_px = owner_distance_px
        self.stationary_radius_px = stationary_radius_px

    def _nearest_person_distance(self, obj_track: Track, store: TrajectoryStore) -> float:
        obj_point = obj_track.points[-1]
        persons = store.by_class(["person"])
        if not persons:
            return math.inf
        return min(
            math.hypot(p.points[-1].x - obj_point.x, p.points[-1].y - obj_point.y)
            for p in persons
            if p.points
        )

    def evaluate(
        self, store: TrajectoryStore, source: str, now: float | None = None
    ) -> list[AnomalyEvent]:
        events: list[AnomalyEvent] = []
        for track in store.tracks.values():
            if track.cls not in self.classes or len(track.points) < 2:
                continue
            eval_now = track.points[-1].timestamp if now is None else now

            radius = track.max_radius_over(self.stationary_sec, now=eval_now)
            duration = track.duration_in_window(self.stationary_sec, now=eval_now)
            if radius > self.stationary_radius_px or duration < self.stationary_sec:
                track.state["abandoned_alerted"] = False
                continue

            owner_distance = self._nearest_person_distance(track, store)
            if owner_distance < self.owner_distance_px:
                track.state["abandoned_alerted"] = False
                continue
            if track.state.get("abandoned_alerted"):
                continue

            track.state["abandoned_alerted"] = True
            latest = track.points[-1]
            confidence = min(1.0, 0.6 + duration / (self.stationary_sec * 2))
            events.append(
                AnomalyEvent(
                    event_type="abandoned_object",
                    track_id=track.track_id,
                    source=source,
                    confidence=confidence,
                    timestamp=eval_now,
                    bbox=(
                        latest.x - latest.w / 2,
                        latest.y - latest.h / 2,
                        latest.x + latest.w / 2,
                        latest.y + latest.h / 2,
                    ),
                    details={
                        "cls_name": track.cls_name,
                        "stationary_sec": round(duration, 1),
                        "owner_distance_px": None
                        if math.isinf(owner_distance)
                        else round(owner_distance, 1),
                    },
                )
            )
        return events
