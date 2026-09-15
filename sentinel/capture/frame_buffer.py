"""Bounded, thread-safe per-source frame buffering: a latest-frame slot for live display/inference
and a rolling time-windowed history used to extract "pre-event" video clips around an alert.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass
class TimestampedFrame:
    timestamp: float
    frame: Any


class FrameBuffer:
    """One instance per source. Producer thread calls `put`; consumers call `get_latest`
    or `get_window` (for clip extraction).
    """

    def __init__(self, max_history_sec: float = 15.0, assumed_fps: float = 15.0) -> None:
        self._lock = threading.Lock()
        self._latest: TimestampedFrame | None = None
        maxlen = max(1, int(max_history_sec * assumed_fps * 1.5))
        self._history: deque[TimestampedFrame] = deque(maxlen=maxlen)
        self.dropped_count = 0

    def put(self, frame: Any) -> None:
        ts = time.time()
        item = TimestampedFrame(ts, frame)
        with self._lock:
            self._latest = item
            self._history.append(item)

    def get_latest(self) -> TimestampedFrame | None:
        with self._lock:
            return self._latest

    def get_window(self, start_ts: float, end_ts: float) -> list[TimestampedFrame]:
        with self._lock:
            return [f for f in self._history if start_ts <= f.timestamp <= end_ts]

    def get_recent_seconds(self, seconds: float) -> list[TimestampedFrame]:
        cutoff = time.time() - seconds
        with self._lock:
            return [f for f in self._history if f.timestamp >= cutoff]
