"""Enables OSNet appearance-ReID inside BoT-SORT.

Ultralytics' bundled `botsort.yaml` ships with `with_reid: False` by default (motion-only
matching). Re-identifying a track by appearance (not just position) is what lets the tracker
survive occlusions -- Challenge 1 in the project spec. This module writes a local copy of the
tracker config with ReID turned on, so `YoloEngine` can point at it via
`tracking.tracker: config/botsort_reid.yaml`.
"""
from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

DEFAULT_REID_MODEL = "osnet_x0_25_msmt17.pt"


def write_reid_tracker_config(
    output_path: str | Path = "config/botsort_reid.yaml",
    reid_model: str = DEFAULT_REID_MODEL,
    base_tracker: str = "botsort.yaml",
) -> Path:
    """Generate a BoT-SORT config with ReID enabled. Requires `pip install -e .[actions]`
    style extras that include `torchreid`/`boxmot` for the ReID backbone; if that dependency
    isn't installed, Ultralytics will raise a clear ImportError at track() time rather than
    failing silently.
    """
    from ultralytics.utils import checks  # noqa: F401  (import guard: confirms ultralytics present)

    try:
        from ultralytics.cfg import TRACKER_CONFIG_DIR

        base_path = Path(TRACKER_CONFIG_DIR) / base_tracker
        cfg = yaml.safe_load(base_path.read_text())
    except Exception:
        logger.warning(
            "Could not locate bundled %s, writing a minimal BoT-SORT+ReID config instead",
            base_tracker,
        )
        cfg = {
            "tracker_type": "botsort",
            "track_high_thresh": 0.5,
            "track_low_thresh": 0.1,
            "new_track_thresh": 0.6,
            "track_buffer": 60,
            "match_thresh": 0.8,
            "fuse_score": True,
            "gmc_method": "sparseOptFlow",
            "proximity_thresh": 0.5,
            "appearance_thresh": 0.25,
        }

    cfg["with_reid"] = True
    cfg["model"] = reid_model

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    logger.info("Wrote ReID-enabled tracker config to %s", output_path)
    return output_path
