"""Thin cv2.VideoCapture wrapper that knows how to (re)open webcam / file / RTSP sources."""
from __future__ import annotations

import logging
import time
from typing import Any

import cv2

logger = logging.getLogger(__name__)


class VideoSource:
    """Opens a webcam index, video file path, or RTSP URL and yields frames.

    Handles reconnection for sources that can drop (webcam unplugged, RTSP timeout).
    Video files intentionally do NOT reconnect on natural end-of-stream (that's just EOF);
    they do reconnect on a genuine read failure mid-stream.
    """

    def __init__(
        self,
        name: str,
        source_type: str,
        source: Any,
        max_retries: int = 10,
        retry_delay_sec: float = 2.0,
        loop_video_files: bool = True,
    ) -> None:
        self.name = name
        self.source_type = source_type
        self.source = source
        self.max_retries = max_retries
        self.retry_delay_sec = retry_delay_sec
        self.loop_video_files = loop_video_files
        self.cap: cv2.VideoCapture | None = None
        self._open()

    def _open(self) -> bool:
        target = int(self.source) if self.source_type == "webcam" else str(self.source)
        cap = cv2.VideoCapture(target)
        if not cap.isOpened():
            logger.warning("Source %s: failed to open %r", self.name, target)
            self.cap = None
            return False
        self.cap = cap
        logger.info("Source %s: opened %r", self.name, target)
        return True

    def read(self) -> tuple[bool, Any]:
        if self.cap is None:
            return False, None
        ok, frame = self.cap.read()
        if ok:
            return True, frame

        if self.source_type == "file" and not self.loop_video_files:
            return False, None

        # Reconnect for webcam/RTSP drops, or loop video files back to the start.
        if self.source_type == "file":
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
            if ok:
                return True, frame

        logger.warning("Source %s: read failed, attempting reconnect", self.name)
        return self._reconnect()

    def _reconnect(self) -> tuple[bool, Any]:
        if self.cap is not None:
            self.cap.release()
        for attempt in range(1, self.max_retries + 1):
            time.sleep(self.retry_delay_sec)
            if self._open():
                ok, frame = self.cap.read()
                if ok:
                    logger.info("Source %s: reconnected on attempt %d", self.name, attempt)
                    return True, frame
            logger.warning(
                "Source %s: reconnect attempt %d/%d failed", self.name, attempt, self.max_retries
            )
        logger.error("Source %s: giving up after %d retries", self.name, self.max_retries)
        return False, None

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    @property
    def fps(self) -> float:
        if self.cap is None:
            return 0.0
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return fps if fps and fps > 0 else 30.0
