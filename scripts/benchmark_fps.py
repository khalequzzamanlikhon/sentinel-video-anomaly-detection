"""Measures actual end-to-end inference FPS on your hardware -- the honest replacement for
assuming "~15 FPS on an RTX 3060" without measuring it. Run this and put the real number in
your README.

Usage:
    python scripts/benchmark_fps.py --source data/raw/sample_clips/sample1.mp4 --frames 200
    python scripts/benchmark_fps.py --source 0 --frames 200  # webcam
"""
from __future__ import annotations

import argparse
import time

import cv2

from sentinel.detection.yolo_engine import YoloEngine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/raw/sample_clips/sample1.mp4")
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--tracker", default="bytetrack.yaml")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source {args.source!r}")

    engine = YoloEngine(weights=args.weights, device=args.device, tracker=args.tracker)

    # Warm up (first inference includes lazy CUDA/model init -- exclude from timing).
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Source produced no frames")
    engine.track(frame)

    n = 0
    t0 = time.time()
    while n < args.frames:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        engine.track(frame)
        n += 1
    elapsed = time.time() - t0
    cap.release()

    fps = n / elapsed if elapsed > 0 else 0.0
    print(f"Processed {n} frames in {elapsed:.2f}s -> {fps:.2f} FPS "
          f"(weights={args.weights}, device={args.device}, tracker={args.tracker})")


if __name__ == "__main__":
    main()
