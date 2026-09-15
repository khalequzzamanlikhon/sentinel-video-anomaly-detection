"""Runs one capture thread per configured source, each writing into its own FrameBuffer."""
from __future__ import annotations

import logging
import threading
import time

from sentinel.capture.frame_buffer import FrameBuffer
from sentinel.capture.rtsp_client import VideoSource
from sentinel.core.config import SourceConfig
from sentinel.core.metrics import FRAMES_PROCESSED

logger = logging.getLogger(__name__)


class StreamManager:
    def __init__(
        self,
        sources: list[SourceConfig],
        max_retries: int = 10,
        retry_delay_sec: float = 2.0,
        clip_buffer_sec: float = 15.0,
    ) -> None:
        self.sources = sources
        self.buffers: dict[str, FrameBuffer] = {
            s.name: FrameBuffer(max_history_sec=clip_buffer_sec) for s in sources
        }
        self._max_retries = max_retries
        self._retry_delay_sec = retry_delay_sec
        self._threads: list[threading.Thread] = []
        self._stop_event = threading.Event()

    def start(self) -> None:
        for src_cfg in self.sources:
            t = threading.Thread(target=self._capture_loop, args=(src_cfg,), daemon=True)
            t.start()
            self._threads.append(t)
        logger.info("StreamManager started %d source thread(s)", len(self._threads))

    def _capture_loop(self, cfg: SourceConfig) -> None:
        video_source = VideoSource(
            name=cfg.name,
            source_type=cfg.type,
            source=cfg.source,
            max_retries=self._max_retries,
            retry_delay_sec=self._retry_delay_sec,
        )
        buffer = self.buffers[cfg.name]
        target_dt = 1.0 / max(video_source.fps, 1.0)
        while not self._stop_event.is_set():
            ok, frame = video_source.read()
            if not ok:
                if cfg.type == "file":
                    logger.info("Source %s: end of stream, stopping thread", cfg.name)
                    break
                logger.error("Source %s: unrecoverable, stopping thread", cfg.name)
                break
            buffer.put(frame)
            FRAMES_PROCESSED.labels(source=cfg.name).inc()
            time.sleep(min(target_dt, 0.05))
        video_source.release()

    def stop(self) -> None:
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=2.0)
        logger.info("StreamManager stopped")

    def get_latest_frames(self) -> dict[str, object]:
        return {name: buf.get_latest() for name, buf in self.buffers.items()}
