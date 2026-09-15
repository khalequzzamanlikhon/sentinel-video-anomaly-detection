"""Generates small synthetic .mp4 clips under data/raw/sample_clips/ so the default
config/cameras.yaml has something to play without requiring you to source real footage first.
These are just moving colored rectangles -- useful for smoke-testing the pipeline (frame I/O,
tracking IDs, dashboard) end to end, NOT for evaluating detection/anomaly accuracy (a synthetic
rectangle won't be classified as "person" by YOLO). Point cameras.yaml at real clips for that.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "sample_clips"


def make_clip(path: Path, seconds: int = 10, fps: int = 15, size: tuple[int, int] = (640, 480)) -> None:
    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n_frames = seconds * fps
    x = 50
    for i in range(n_frames):
        frame = np.full((h, w, 3), 30, dtype=np.uint8)
        x = (x + 4) % (w - 80)
        cv2.rectangle(frame, (x, h // 2 - 60), (x + 80, h // 2 + 60), (0, 180, 255), -1)
        cv2.putText(frame, f"frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    make_clip(OUT_DIR / "sample1.mp4")
    make_clip(OUT_DIR / "sample2.mp4")
    print(f"Wrote sample clips to {OUT_DIR}")


if __name__ == "__main__":
    main()
