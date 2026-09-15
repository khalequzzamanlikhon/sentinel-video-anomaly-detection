"""Builds a short per-track video clip (a stack of cropped frames) from a source's rolling
frame buffer, for action recognition. Maintaining a per-track circular buffer of raw pixels
(rather than re-cropping from the source buffer every time) is what the original spec's
Challenge 4 calls for; we get the same effect more simply by re-cropping from the already-kept
`FrameBuffer` history using each track's bbox at each timestamp, which avoids duplicating
frame storage per track.
"""
from __future__ import annotations

import cv2
import numpy as np

from sentinel.capture.frame_buffer import FrameBuffer
from sentinel.tracking.trajectory_store import Track


def extract_track_clip(
    track: Track,
    buffer: FrameBuffer,
    clip_len_sec: float = 2.0,
    crop_size: int = 224,
    min_frames: int = 8,
) -> np.ndarray | None:
    """Returns an (T, H, W, 3) uint8 array (RGB) cropped around the track's bbox at each
    timestamp, or None if there isn't enough history yet (buffer or track too short --
    Challenge 4: a track visible for only ~1s before occlusion won't produce a clip).
    """
    if not track.points:
        return None
    latest_ts = track.points[-1].timestamp
    frame_window = buffer.get_window(latest_ts - clip_len_sec, latest_ts)
    if len(frame_window) < min_frames:
        return None

    crops = []
    for tf in frame_window:
        # nearest track point in time to this frame
        pt = min(track.points, key=lambda p: abs(p.timestamp - tf.timestamp))
        x1 = max(int(pt.x - pt.w / 2), 0)
        y1 = max(int(pt.y - pt.h / 2), 0)
        x2 = min(int(pt.x + pt.w / 2), tf.frame.shape[1])
        y2 = min(int(pt.y + pt.h / 2), tf.frame.shape[0])
        if x2 <= x1 or y2 <= y1:
            continue
        crop = tf.frame[y1:y2, x1:x2]
        crop = cv2.resize(crop, (crop_size, crop_size))
        crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

    if len(crops) < min_frames:
        return None
    return np.stack(crops, axis=0)
