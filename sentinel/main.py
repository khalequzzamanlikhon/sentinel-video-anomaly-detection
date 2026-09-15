"""CLI entry point: `python -m sentinel.main` runs the pipeline and shows a live OpenCV
window per source with bounding boxes, track IDs, and alert banners.
"""
from __future__ import annotations

import argparse
import logging
import time

import cv2

from sentinel.core.config import load_config
from sentinel.core.metrics import start_metrics_server
from sentinel.core.pipeline import SentinelPipeline
from sentinel.core.render import draw_overlay

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sentinel video anomaly detection pipeline")
    parser.add_argument("--config", default="config", help="Directory containing YAML configs")
    parser.add_argument("--headless", action="store_true", help="Don't open display windows")
    parser.add_argument("--metrics-port", type=int, default=8000)
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = SentinelPipeline(config)
    start_metrics_server(args.metrics_port)
    pipeline.start()

    logger.info("Sentinel running on %d source(s). Press 'q' to quit.", len(config.sources))
    try:
        while True:
            results = pipeline.step()
            if not args.headless:
                for name, res in results.items():
                    if res["frame"] is None:
                        continue
                    frame = draw_overlay(res["frame"].copy(), res["detections"], res["events"])
                    cv2.imshow(name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                time.sleep(0.01)
    except KeyboardInterrupt:
        logger.info("Interrupted, shutting down")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
