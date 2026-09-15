"""Long-running soak test: runs the pipeline for a configurable duration and reports
crashes, FPS stability, memory growth, and alert counts. Use this to actually produce the
"99% uptime over 24h" style numbers instead of assuming them -- run it, then put the real
result in your README.

Usage:
    python scripts/stress_test.py --duration 1h --config config
    python scripts/stress_test.py --duration 8h --config config --headless
"""
from __future__ import annotations

import argparse
import logging
import re
import time

from sentinel.core.config import load_config
from sentinel.core.pipeline import SentinelPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_duration(s: str) -> float:
    m = re.match(r"^(\d+(?:\.\d+)?)(s|m|h)$", s.strip())
    if not m:
        raise ValueError(f"Invalid duration {s!r}, expected e.g. '30s', '10m', '8h'")
    value, unit = float(m.group(1)), m.group(2)
    return value * {"s": 1, "m": 60, "h": 3600}[unit]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", default="10m", help="e.g. 30s, 10m, 8h")
    parser.add_argument("--config", default="config")
    parser.add_argument("--sample-interval-sec", type=float, default=30.0)
    args = parser.parse_args()

    try:
        import psutil

        process = psutil.Process()
    except ImportError:
        process = None
        logger.warning("psutil not installed -- memory growth won't be tracked (pip install psutil)")

    duration_sec = parse_duration(args.duration)
    config = load_config(args.config)
    pipeline = SentinelPipeline(config)
    pipeline.start()

    start = time.time()
    last_sample = start
    total_alerts = 0
    total_steps = 0
    crashes = 0
    mem_samples = []

    logger.info("Starting %s soak test on %d source(s)", args.duration, len(config.sources))
    try:
        while time.time() - start < duration_sec:
            try:
                results = pipeline.step()
                total_alerts += sum(len(r["events"]) for r in results.values())
                total_steps += 1
            except Exception:
                crashes += 1
                logger.exception("Pipeline step raised an exception (count=%d)", crashes)

            if time.time() - last_sample >= args.sample_interval_sec:
                last_sample = time.time()
                elapsed_min = (time.time() - start) / 60
                mem_mb = process.memory_info().rss / 1e6 if process else None
                if mem_mb is not None:
                    mem_samples.append(mem_mb)
                logger.info(
                    "t=%.1fmin steps=%d alerts=%d crashes=%d mem=%s",
                    elapsed_min, total_steps, total_alerts, crashes,
                    f"{mem_mb:.1f}MB" if mem_mb is not None else "n/a",
                )
            time.sleep(0.01)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        pipeline.stop()

    elapsed = time.time() - start
    print("\n--- Soak test summary ---")
    print(f"Duration: {elapsed / 60:.1f} min")
    print(f"Steps processed: {total_steps}")
    print(f"Alerts generated: {total_alerts}")
    print(f"Exceptions caught: {crashes}")
    if mem_samples:
        print(f"Memory: start={mem_samples[0]:.1f}MB end={mem_samples[-1]:.1f}MB "
              f"growth={mem_samples[-1] - mem_samples[0]:+.1f}MB")
    uptime_pct = 100.0 if crashes == 0 else max(0.0, 100.0 * (1 - crashes / max(total_steps, 1)))
    print(f"Approx uptime: {uptime_pct:.2f}%")


if __name__ == "__main__":
    main()
