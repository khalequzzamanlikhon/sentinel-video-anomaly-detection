"""YOLOv8 detection + tracking engine.

Ultralytics' `model.track()` performs detection and multi-object tracking (ByteTrack or
BoT-SORT, optionally with ReID) in a single call and returns persistent track IDs, so
detection and tracking are wired together here rather than as two separate inference passes.
`sentinel/tracking/` consumes the resulting boxes to build trajectory history and does not
reimplement the tracker itself -- see `sentinel/tracking/bot_sort_wrapper.py` for why.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    track_id: int | None
    cls: int
    cls_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @property
    def width(self) -> float:
        return self.xyxy[2] - self.xyxy[0]

    @property
    def height(self) -> float:
        return self.xyxy[3] - self.xyxy[1]


class YoloEngine:
    def __init__(
        self,
        weights: str = "yolov8n.pt",
        confidence: float = 0.35,
        iou: float = 0.5,
        device: str = "auto",
        classes: list[int] | None = None,
        tracker: str = "botsort.yaml",
        persist: bool = True,
    ) -> None:
        from ultralytics import YOLO  # lazy import: heavy dep, only needed when actually detecting

        self.model = YOLO(weights)
        self.confidence = confidence
        self.iou = iou
        self.device = None if device == "auto" else device
        self.classes = classes
        self.tracker = tracker
        self.persist = persist
        self.names = self.model.names

    def track(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.track(
            frame,
            conf=self.confidence,
            iou=self.iou,
            device=self.device,
            classes=self.classes,
            tracker=self.tracker,
            persist=self.persist,
            verbose=False,
        )
        return self._parse(results)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detection-only pass, no tracking (used by benchmarks / tests)."""
        results = self.model.predict(
            frame,
            conf=self.confidence,
            iou=self.iou,
            device=self.device,
            classes=self.classes,
            verbose=False,
        )
        return self._parse(results)

    def _parse(self, results: Any) -> list[Detection]:
        out: list[Detection] = []
        if not results:
            return out
        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return out
        ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(boxes)
        xyxy = boxes.xyxy.tolist()
        confs = boxes.conf.tolist()
        clss = boxes.cls.int().tolist()
        for track_id, box, conf, cls in zip(ids, xyxy, confs, clss):
            out.append(
                Detection(
                    track_id=track_id,
                    cls=cls,
                    cls_name=self.names.get(cls, str(cls)),
                    confidence=conf,
                    xyxy=tuple(box),
                )
            )
        return out
