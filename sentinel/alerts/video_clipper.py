"""Extracts a thumbnail + a short video clip (pre/post event window) from a source's rolling
frame buffer once an alert fires.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2

from sentinel.anomaly.events import AnomalyEvent
from sentinel.capture.frame_buffer import FrameBuffer

logger = logging.getLogger(__name__)


def _closest_frame(buffer: FrameBuffer, timestamp: float):
    latest = buffer.get_latest()
    if latest is None:
        return None
    window = buffer.get_recent_seconds(seconds=max(30.0, abs(latest.timestamp - timestamp) + 5))
    if not window:
        return None
    return min(window, key=lambda f: abs(f.timestamp - timestamp))


def save_thumbnail(buffer: FrameBuffer, event: AnomalyEvent, thumbnails_dir: str | Path) -> str | None:
    frame = _closest_frame(buffer, event.timestamp)
    if frame is None:
        return None
    thumbnails_dir = Path(thumbnails_dir)
    thumbnails_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{event.source}_{event.event_type}_{int(event.timestamp)}_{event.track_id}.jpg"
    path = thumbnails_dir / filename

    img = frame.frame.copy()
    if event.bbox:
        x1, y1, x2, y2 = (int(v) for v in event.bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            img, event.event_type, (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
        )
    cv2.imwrite(str(path), img)
    return str(path)


def extract_clip(
    buffer: FrameBuffer,
    event: AnomalyEvent,
    clips_dir: str | Path,
    pre_event_sec: float = 5.0,
    post_event_sec: float = 5.0,
    fps: float = 15.0,
) -> str | None:
    """Call this once at least `post_event_sec` has elapsed since `event.timestamp` so the
    buffer actually contains the post-event frames (see AlertManager, which schedules this).
    """
    frames = buffer.get_window(event.timestamp - pre_event_sec, event.timestamp + post_event_sec)
    if len(frames) < 2:
        logger.warning("Not enough buffered frames to build a clip for event %s", event.event_type)
        return None

    clips_dir = Path(clips_dir)
    clips_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{event.source}_{event.event_type}_{int(event.timestamp)}_{event.track_id}.mp4"
    path = clips_dir / filename

    h, w = frames[0].frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    try:
        for f in frames:
            writer.write(f.frame)
    finally:
        writer.release()
    return str(path)
