"""Offline Sentinel demo renderer for README assets.

sentinel.main --headless doesn't save video, and its capture threads pace to wall-clock and
drop frames. This drives the same YoloEngine (YOLOv8 + BoT-SORT), TrajectoryStore, RuleEngine
and draw_overlay frame-by-frame on *video time*, then writes an annotated H.264 MP4, a GIF,
alert thumbnails and summary.json.
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import time
from collections import Counter
from pathlib import Path

import cv2

from sentinel.alerts.alert_manager import DEFAULT_SEVERITY
from sentinel.anomaly.rule_engine import RuleEngine
from sentinel.core.config import load_config
from sentinel.core.render import draw_overlay
from sentinel.detection.yolo_engine import YoloEngine
from sentinel.tracking.trajectory_store import TrajectoryStore

# The shipped thresholds (30 s loitering / crowd, 5 min abandoned bag) can't fire inside a
# ~1 minute sample clip, so the "demo" profile scales the time windows down.
DEMO_RULE_OVERRIDES = {
    "loitering": {"min_duration_sec": 6, "radius_px": 50},
    "crowd_density": {"person_count_threshold": 6, "sustained_sec": 3},
    "wrong_way": {"min_persist_sec": 1.5},
    "abandoned_object": {"stationary_sec": 10},
}
BANNER_SEC = 2.5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="config")
    ap.add_argument("--weights", default=None, help="defaults to config/models.yaml")
    ap.add_argument("--device", default="0")
    ap.add_argument("--profile", choices=["default", "demo"], default="demo")
    ap.add_argument("--flow-deg", type=float, default=0.0)
    ap.add_argument("--gif-seconds", type=float, default=12.0)
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    det_cfg = cfg.models.get("detection", {})
    track_cfg = cfg.models.get("tracking", {})
    pipe_cfg = cfg.models.get("pipeline", {})
    alerting = cfg.alerts.get("alerting", {})
    severity_map = {**DEFAULT_SEVERITY, **alerting.get("severity", {})}
    dedup_window = alerting.get("dedup_window_sec", 30)

    rules = copy.deepcopy(cfg.alerts.get("rules", {}))
    if args.profile == "demo":
        for name, override in DEMO_RULE_OVERRIDES.items():
            rules.setdefault(name, {}).update(override)

    weights = args.weights or det_cfg.get("weights", "yolov8n.pt")
    engine = YoloEngine(
        weights=weights,
        confidence=det_cfg.get("confidence", 0.35),
        iou=det_cfg.get("iou", 0.5),
        device=args.device,
        classes=det_cfg.get("classes"),
        tracker=track_cfg.get("tracker", "botsort.yaml"),
        persist=True,
    )
    store = TrajectoryStore(history_len=pipe_cfg.get("trajectory_history_len", 150))
    rule_engine = RuleEngine(rules)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = Path(args.out)
    (out / "thumbnails").mkdir(parents=True, exist_ok=True)
    raw_path = out / "annotated_raw.avi"
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

    last_alert: dict[tuple, float] = {}
    alerts: list[dict] = []
    banners: list[tuple[float, object]] = []
    seen_tracks: set[tuple[str, int]] = set()
    infer_s: list[float] = []
    max_persons = 0
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and idx >= args.max_frames):
            break
        t = idx / fps

        t0 = time.perf_counter()
        detections = engine.track(frame)
        infer_s.append(time.perf_counter() - t0)

        for det in detections:
            if det.track_id is None:
                continue
            track = store.update(det.track_id, det.cls, det.cls_name, det.xyxy, det.confidence, t)
            if track.first_seen > 1e6:  # Track defaults first_seen to wall-clock time.time()
                track.first_seen = t
            seen_tracks.add((det.cls_name, det.track_id))
        store.prune_stale(t)
        persons = sum(1 for det in detections if det.cls_name == "person")
        max_persons = max(max_persons, persons)

        events = rule_engine.evaluate(store, "demo", expected_flow_deg=args.flow_deg, frame_shape=(h, w), now=t)
        for event in events:
            key = (event.event_type, event.track_id)
            if key in last_alert and t - last_alert[key] < dedup_window:
                continue
            last_alert[key] = t
            thumb_name = f"{idx:05d}_{event.event_type}_{event.track_id}.jpg"
            thumb = frame.copy()
            if event.bbox:
                x1, y1, x2, y2 = (int(v) for v in event.bbox)
                cv2.rectangle(thumb, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(thumb, event.event_type, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imwrite(str(out / "thumbnails" / thumb_name), thumb)
            alerts.append({
                "frame": idx,
                "video_time_s": round(t, 2),
                "event_type": event.event_type,
                "severity": severity_map.get(event.event_type, "low"),
                "track_id": event.track_id,
                "confidence": round(event.confidence, 3),
                "details": event.details,
                "thumbnail": thumb_name,
            })
            banners.append((t + BANNER_SEC, event))
            print(f"[{t:6.2f}s] ALERT {event.event_type} track={event.track_id} conf={event.confidence:.2f} {event.details}")

        banners = [(until, e) for until, e in banners if until >= t]
        vis = draw_overlay(frame.copy(), detections, [e for _, e in banners])
        hud = f"t={t:5.1f}s  persons={persons}  alerts={len(alerts)}  {1.0 / max(infer_s[-1], 1e-6):.0f} FPS"
        cv2.putText(vis, hud, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        writer.write(vis)
        idx += 1

    cap.release()
    writer.release()

    mp4 = out / "annotated.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_path), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart", str(mp4)], check=True)
    raw_path.unlink()
    if args.gif_seconds > 0:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-t", str(args.gif_seconds), "-i", str(mp4), "-vf",
                        "fps=10,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse",
                        str(out / "demo.gif")], check=True)

    steady = infer_s[5:] or infer_s
    mean_ms = 1000.0 * sum(steady) / max(len(steady), 1)
    summary = {
        "source": str(args.source),
        "profile": args.profile,
        "weights": weights,
        "tracker": track_cfg.get("tracker", "botsort.yaml"),
        "frames": idx,
        "video_fps": fps,
        "resolution": [w, h],
        "mean_track_ms": round(mean_ms, 2),
        "track_fps": round(1000.0 / mean_ms, 1) if mean_ms else None,
        "unique_tracks_by_class": dict(Counter(cls for cls, _ in seen_tracks)),
        "max_persons_in_frame": max_persons,
        "alerts_by_type": dict(Counter(a["event_type"] for a in alerts)),
        "alerts": alerts,
        "rules": rules,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("alerts", "rules")}, indent=2))


if __name__ == "__main__":
    main()
