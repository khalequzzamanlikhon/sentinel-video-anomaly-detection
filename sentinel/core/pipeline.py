"""The main per-frame processing loop: pulls the latest frame from each source, runs
detection+tracking, updates trajectory stores, runs the rule engine, and hands events to the
alert manager. Frame skipping (`process_every_nth_frame`) trades detection frequency for
throughput, per Challenge 3 in the spec -- skipped frames are still drawn (for live display)
using the last known boxes, just not re-detected.
"""
from __future__ import annotations

import logging
import time

from sentinel.alerts.alert_manager import AlertManager
from sentinel.anomaly.rule_engine import RuleEngine
from sentinel.capture.stream_manager import StreamManager
from sentinel.core.config import SentinelConfig
from sentinel.core.metrics import INFERENCE_FPS, STREAM_LAG_SECONDS
from sentinel.detection.yolo_engine import YoloEngine
from sentinel.tracking.trajectory_store import TrajectoryStore

logger = logging.getLogger(__name__)


class SentinelPipeline:
    def __init__(self, config: SentinelConfig) -> None:
        self.config = config
        self.sources = config.sources

        det_cfg = config.models.get("detection", {})
        track_cfg = config.models.get("tracking", {})
        self.pipeline_cfg = config.models.get("pipeline", {})

        self.engine = YoloEngine(
            weights=det_cfg.get("weights", "yolov8n.pt"),
            confidence=det_cfg.get("confidence", 0.35),
            iou=det_cfg.get("iou", 0.5),
            device=det_cfg.get("device", "auto"),
            classes=det_cfg.get("classes"),
            tracker=track_cfg.get("tracker", "botsort.yaml"),
            persist=track_cfg.get("persist", True),
        )

        history_len = self.pipeline_cfg.get("trajectory_history_len", 150)
        self.stores: dict[str, TrajectoryStore] = {
            s.name: TrajectoryStore(history_len=history_len) for s in self.sources
        }

        self.rule_engine = RuleEngine(config.alerts.get("rules", {}))

        reconnect_cfg = config.cameras.get("reconnect", {})
        self.stream_manager = StreamManager(
            self.sources,
            max_retries=reconnect_cfg.get("max_retries", 10),
            retry_delay_sec=reconnect_cfg.get("retry_delay_sec", 2.0),
        )

        self.alert_manager = AlertManager(config.alerts, self.stream_manager.buffers)

        self.action_classifier = None
        actions_cfg = config.models.get("actions", {})
        if actions_cfg.get("enabled", False):
            try:
                from sentinel.actions.action_classifier import ActionClassifier

                weights = actions_cfg.get("weights", "pretrained")
                self.action_classifier = ActionClassifier(
                    backend=actions_cfg.get("backend", "x3d"),
                    clip_len_sec=actions_cfg.get("clip_len_sec", 2.0),
                    run_every_sec=actions_cfg.get("run_every_sec", 2.0),
                    weights_path=None if weights == "pretrained" else weights,
                )
                logger.info("Action recognition enabled (%s)", actions_cfg.get("backend", "x3d"))
            except ImportError:
                logger.warning(
                    "actions.enabled=true but the 'actions' extra isn't installed "
                    "(pip install -e \".[actions]\"); continuing without action recognition"
                )

        self._frame_counters: dict[str, int] = {s.name: 0 for s in self.sources}
        self._fps_window: dict[str, list[float]] = {s.name: [] for s in self.sources}
        self._running = False

    def start(self) -> None:
        self.stream_manager.start()
        self._running = True

    def stop(self) -> None:
        self._running = False
        self.stream_manager.stop()
        self.alert_manager.shutdown()

    def step(self) -> dict[str, dict]:
        """Process one round of frames (one per source). Returns per-source results:
        {"frame": ndarray|None, "detections": [...], "events": [...]}. Intended to be called
        in a loop by `main.py` or the Streamlit dashboard.
        """
        results: dict[str, dict] = {}
        nth = max(1, self.pipeline_cfg.get("process_every_nth_frame", 1))

        for src in self.sources:
            buf = self.stream_manager.buffers[src.name]
            latest = buf.get_latest()
            if latest is None:
                results[src.name] = {"frame": None, "detections": [], "events": []}
                continue

            STREAM_LAG_SECONDS.labels(source=src.name).set(max(0.0, time.time() - latest.timestamp))

            self._frame_counters[src.name] += 1
            should_process = self._frame_counters[src.name] % nth == 0

            detections = []
            events = []
            if should_process:
                t0 = time.time()
                detections = self.engine.track(latest.frame)
                store = self.stores[src.name]
                for det in detections:
                    if det.track_id is None:
                        continue
                    store.update(det.track_id, det.cls, det.cls_name, det.xyxy, det.confidence, latest.timestamp)
                store.prune_stale(latest.timestamp)

                h, w = latest.frame.shape[:2]
                events = self.rule_engine.evaluate(
                    store, src.name, expected_flow_deg=src.expected_flow_deg, frame_shape=(h, w)
                )
                if events:
                    self.alert_manager.process(events)

                if self.action_classifier is not None:
                    for track in store.by_class(["person"]):
                        self.action_classifier.maybe_classify(track, buf)

                elapsed = time.time() - t0
                self._track_fps(src.name, elapsed)

            results[src.name] = {"frame": latest.frame, "detections": detections, "events": events}

        return results

    def _track_fps(self, source: str, elapsed_sec: float) -> None:
        window = self._fps_window[source]
        window.append(elapsed_sec)
        if len(window) > 30:
            window.pop(0)
        avg = sum(window) / len(window)
        fps = 1.0 / avg if avg > 1e-6 else 0.0
        INFERENCE_FPS.labels(source=source).set(fps)
