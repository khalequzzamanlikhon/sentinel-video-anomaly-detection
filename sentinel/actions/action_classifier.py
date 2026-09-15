"""Orchestrates action recognition on active tracks: builds a clip, runs the backend, and
caches the result for `run_every_sec` so we don't re-run inference every frame (per spec
Phase 2 / Challenge 4).
"""
from __future__ import annotations

import logging
import time

from sentinel.actions.clip_extractor import extract_track_clip
from sentinel.actions.slowfast_engine import ActionPrediction, SlowFastEngine
from sentinel.capture.frame_buffer import FrameBuffer
from sentinel.tracking.trajectory_store import Track

logger = logging.getLogger(__name__)


class ActionClassifier:
    def __init__(
        self,
        backend: str = "x3d",
        clip_len_sec: float = 2.0,
        run_every_sec: float = 2.0,
        weights_path: str | None = None,
        device: str = "cpu",
    ) -> None:
        self.engine = SlowFastEngine(backend=backend, device=device, weights_path=weights_path)
        self.clip_len_sec = clip_len_sec
        self.run_every_sec = run_every_sec

    def maybe_classify(self, track: Track, buffer: FrameBuffer) -> ActionPrediction | None:
        now = time.time()
        last_run = track.state.get("action_last_run", 0.0)
        if now - last_run < self.run_every_sec:
            return track.state.get("action_last_result")

        clip = extract_track_clip(track, buffer, clip_len_sec=self.clip_len_sec)
        if clip is None:
            return track.state.get("action_last_result")

        try:
            preds = self.engine.predict(clip, top_k=1)
        except Exception:
            logger.exception("Action recognition failed for track %s", track.track_id)
            return track.state.get("action_last_result")

        track.state["action_last_run"] = now
        result = preds[0] if preds else None
        track.state["action_last_result"] = result
        if result:
            logger.info(
                "Track %s: %s, action=%s, confidence=%.2f",
                track.track_id, track.cls_name, result.label, result.confidence,
            )
        return result
