"""Tracker configuration resolution.

Design note: BoT-SORT (Kalman-filter motion model + Hungarian matching + OSNet ReID for
appearance matching through occlusions) is implemented and maintained inside
`ultralytics/trackers/`. `YoloEngine.track()` (see `sentinel/detection/yolo_engine.py`) drives
it via `model.track(tracker=..., persist=True)`. Reimplementing BoT-SORT's Kalman filter and
cascaded matching from scratch would not produce a more correct tracker than the maintained
implementation, so this module's job is just to validate/resolve tracker config -- the actual
tracking math lives in the `ultralytics` dependency.

To use ByteTrack instead (faster, no appearance ReID, worse through long occlusions), set
`tracking.tracker: bytetrack.yaml` in `config/models.yaml`.
"""
from __future__ import annotations

VALID_TRACKERS = {"botsort.yaml", "bytetrack.yaml"}


def resolve_tracker_config(tracker: str) -> str:
    if tracker not in VALID_TRACKERS:
        raise ValueError(f"Unknown tracker {tracker!r}, expected one of {VALID_TRACKERS}")
    return tracker
