"""Composes the individual rule-based detectors into one call per frame."""
from __future__ import annotations

from sentinel.anomaly.abandoned_object import AbandonedObjectDetector
from sentinel.anomaly.crowd_analyzer import CrowdAnalyzer
from sentinel.anomaly.events import AnomalyEvent
from sentinel.anomaly.fall_detector import FallDetector
from sentinel.anomaly.loitering_detector import LoiteringDetector
from sentinel.anomaly.wrongway_detector import WrongWayDetector
from sentinel.tracking.trajectory_store import TrajectoryStore


class RuleEngine:
    def __init__(self, rules_config: dict, roi_config: dict | None = None) -> None:
        self.rules_config = rules_config
        self.roi_config = roi_config or {}

        fall_cfg = rules_config.get("fall", {})
        self.fall_detector = FallDetector(
            aspect_ratio_drop_pct=fall_cfg.get("aspect_ratio_drop_pct", 50),
            window_sec=fall_cfg.get("window_sec", 0.5),
            min_persist_sec=fall_cfg.get("min_persist_sec", 0.5),
            velocity_threshold_px_per_sec=fall_cfg.get("velocity_threshold_px_per_sec", 40),
        )
        self.fall_enabled = fall_cfg.get("enabled", True)

        loiter_cfg = rules_config.get("loitering", {})
        self.loitering_detector = LoiteringDetector(
            radius_px=loiter_cfg.get("radius_px", 60),
            min_duration_sec=loiter_cfg.get("min_duration_sec", 30),
        )
        self.loitering_enabled = loiter_cfg.get("enabled", True)

        abandoned_cfg = rules_config.get("abandoned_object", {})
        self.abandoned_detector = AbandonedObjectDetector(
            classes=abandoned_cfg.get("classes", [24, 26, 28]),
            stationary_sec=abandoned_cfg.get("stationary_sec", 300),
            owner_distance_px=abandoned_cfg.get("owner_distance_px", 400),
            stationary_radius_px=abandoned_cfg.get("stationary_radius_px", 20),
        )
        self.abandoned_enabled = abandoned_cfg.get("enabled", True)

        crowd_cfg = rules_config.get("crowd_density", {})
        self.crowd_analyzer = CrowdAnalyzer(
            person_count_threshold=crowd_cfg.get("person_count_threshold", 8),
            sustained_sec=crowd_cfg.get("sustained_sec", 30),
        )
        self.crowd_roi = crowd_cfg.get("roi")
        self.crowd_enabled = crowd_cfg.get("enabled", True)

        wrongway_cfg = rules_config.get("wrong_way", {})
        self.wrongway_detector = WrongWayDetector(
            angle_threshold_deg=wrongway_cfg.get("angle_threshold_deg", 90),
            min_speed_px_per_sec=wrongway_cfg.get("min_speed_px_per_sec", 15),
            min_persist_sec=wrongway_cfg.get("min_persist_sec", 2.0),
        )
        self.wrongway_enabled = wrongway_cfg.get("enabled", True)

    def evaluate(
        self,
        store: TrajectoryStore,
        source: str,
        expected_flow_deg: float = 0.0,
        frame_shape: tuple[int, int] | None = None,
        now: float | None = None,
    ) -> list[AnomalyEvent]:
        events: list[AnomalyEvent] = []

        for track in list(store.tracks.values()):
            if self.fall_enabled:
                e = self.fall_detector.evaluate(track, source, now=now)
                if e:
                    events.append(e)
            if self.loitering_enabled:
                e = self.loitering_detector.evaluate(track, source, now=now)
                if e:
                    events.append(e)
            if self.wrongway_enabled:
                e = self.wrongway_detector.evaluate(track, source, expected_flow_deg, now=now)
                if e:
                    events.append(e)

        if self.abandoned_enabled:
            events.extend(self.abandoned_detector.evaluate(store, source, now=now))

        if self.crowd_enabled:
            e = self.crowd_analyzer.evaluate(
                store, source, roi_norm=self.crowd_roi, frame_shape=frame_shape, now=now
            )
            if e:
                events.append(e)

        return events
