# Sentinel — Revised Project Plan

This revises `project_sentinel_anomaly_detection.pdf` (the original "Sentinel" spec) into a plan that
a solo developer can actually **build, run, and demo** without a lab's worth of infrastructure. The
original spec is a great *north star* — this document splits it into what's implemented **now**
(Tier 1, in this repo, working today) and what's a **documented stretch roadmap** (Tier 2/3) that
would need real GPU time, curated datasets, and edge hardware the original 6–8 week/2-person plan
assumed.

## Why rescope?

The original spec bundles four hard projects into one:
1. A real-time multi-stream CV pipeline (detection + tracking) — **very achievable solo**.
2. Rule-based behavior/anomaly detection on top of tracks — **very achievable solo**.
3. Custom-trained action recognition (fine-tuned SlowFast/X3D) + a trained LSTM anomaly
   autoencoder — **needs curated video datasets and GPU training time** that don't exist yet
   (Phase 0's "download MOT17/UCF-101/Kinetics, label 100 clips in Label Studio" is itself a
   multi-day task before any model code runs).
4. Production MLOps + edge deployment (TensorRT, Jetson Nano, Prometheus+Grafana, MLflow+DVC,
   an 8-hour soak test) — **needs real hardware and a long-running environment**, not a coding
   session.

Building (3) and (4) "for real" inside one sitting would mean either faking results (fabricated
mAP/MOTA numbers, a model that was never actually fine-tuned) or shipping broken stubs dressed up
as done. Neither is good for a portfolio — an interviewer who asks "walk me through your MOTA eval"
will find out immediately if the number is invented.

Instead:

- **Tier 1 (this repo, fully working today):** multi-source ingestion, YOLOv8 detection,
  BoT-SORT/ByteTrack tracking with persistent IDs, trajectory storage, a real rule-based anomaly
  engine (falls, loitering, abandoned objects, crowd density, wrong-way movement) with temporal
  consistency + multi-signal fusion, an alert pipeline (dedup, severity, thumbnails, video clips,
  SQLite storage), a Streamlit dashboard, Prometheus metrics, tests, and Docker packaging. All of
  this runs on a pretrained COCO model — **no training data required** to get a working demo.
- **Tier 2 (scaffolded, real code, needs your data/GPU to finish):** an action-recognition plugin
  (pretrained SlowFast/X3D via PyTorchVideo, zero-shot Kinetics-400 labels out of the box, with a
  fine-tuning script for your own 5 custom classes once you've collected clips), an LSTM trajectory
  autoencoder training script, ONNX export. The code is real and runs — the *fine-tuning on your
  data* is the part only you can do, because only you can collect/label the clips.
- **Tier 3 (roadmap, not attempted here):** TensorRT conversion, Jetson Nano/Coral edge deployment,
  MLflow+DVC servers, Grafana dashboards, the formal MOT17/COCO benchmark numbers, the 8-hour soak
  test, the demo video and blog post. These need hardware, time, and content creation that belong
  to you, not a single coding session. [ROADMAP.md](ROADMAP.md) lists exact next steps for each.

This keeps every claim in the README honest: nothing is a fabricated benchmark, and everything
labeled "done" actually runs.

## What changed vs. the original spec

| Original | Revised | Why |
|---|---|---|
| Hand-roll BoT-SORT + OSNet ReID from scratch | Use Ultralytics' built-in BoT-SORT/ByteTrack tracker (`persist=True`, `tracker="botsort.yaml"`), with ReID enabled via config | Ultralytics ships a maintained, correct BoT-SORT+ReID implementation; reimplementing Kalman filters/Hungarian matching from zero adds weeks and a large bug surface for no portfolio value beyond "I typed it myself" |
| Fine-tune SlowFast on 100 custom clips before anything works | Ship a pluggable action-recognition module using pretrained Kinetics-400 weights (zero-shot) by default; fine-tuning script provided, run when you have clips | Removes a hard blocking dependency (you can demo today); fine-tuning remains available as documented follow-up |
| asyncio + multiprocessing pipeline across separate processes | Threaded capture (one thread per source) + a single-process batched inference loop | Simpler to run, debug, and containerize on a laptop; multiprocessing is a documented Tier 3 optimization once FPS is actually the bottleneck |
| TensorRT + Jetson Nano/Coral edge deployment in Week 5–6 | ONNX export script (real) now; TensorRT/Jetson documented in [ROADMAP.md](ROADMAP.md) | Needs an actual Jetson device to validate; shipping untested "TensorRT wrapper" code would be theater |
| MLflow + DVC + Prometheus + Grafana, all wired | Prometheus metrics endpoint wired for real (it's cheap and self-contained); MLflow/DVC hooks added to training scripts behind a config flag; Grafana/DVC-server setup documented | Prometheus needs no external service to demo (`/metrics` endpoint); MLflow/DVC need a server or remote storage that's a per-user setup choice |
| Success criteria: MOTA > 0.75 on MOT17, mAP > 0.65, etc. | Success criteria reframed as: system runs end-to-end on 3 concurrent sources, rule-based alerts fire correctly on staged test clips, FPS is measured and reported for your hardware | You cannot claim a MOTA/mAP number without actually running the official eval harness against the official dataset splits — [ROADMAP.md](ROADMAP.md) shows exactly how to produce those numbers later and drop them into the README once you have |
| Single 6–8 week plan, no clear "done" line | Phases 0–4 = Tier 1, buildable now; Phases 5–6 = Tier 2/3, explicit follow-up | Gives a real stopping point for "portfolio-ready v1" instead of an open-ended research project |

## Implemented in this repo (Tier 1)

- **Ingestion** (`sentinel/capture/`): webcam, video file, and RTSP sources, each on its own
  capture thread, with an auto-reconnect loop for RTSP/USB drops and a bounded frame queue so a
  slow consumer can't leak memory.
- **Detection** (`sentinel/detection/`): Ultralytics YOLOv8 wrapper, configurable model
  size/classes/confidence via `config/models.yaml`.
- **Tracking** (`sentinel/tracking/`): thin wrapper around Ultralytics' persistent tracker
  (ByteTrack or BoT-SORT+ReID, chosen in config) + a per-track trajectory store (bounded history of
  position/size/timestamp) used by every downstream rule.
- **Anomaly rules** (`sentinel/anomaly/`): fall detection (aspect-ratio collapse + vertical
  velocity spike), loitering (bounded-radius dwell time), abandoned object (stationary bag +
  owner-distance), crowd density (ROI occupancy over time), wrong-way movement (velocity angle vs.
  configured flow direction) — each requires the anomaly to persist for a configurable window
  before firing, per the original spec's Challenge 2 mitigation.
- **Alerts** (`sentinel/alerts/`): deduplication (same track+event within a cooldown window),
  severity scoring, thumbnail + rolling video-clip extraction, SQLite persistence.
- **Dashboard** (`sentinel/dashboard/`): Streamlit app — live grid view, alert feed with
  thumbnails and clip playback, analytics charts, CSV export.
- **Metrics** (`sentinel/core/metrics.py`): real Prometheus counters/gauges
  (`inference_fps`, `alert_count`, `stream_lag_seconds`) exposed over HTTP.
- **Tests** (`tests/`): pure-logic unit tests for the rule engine, trajectory store, and alert
  dedup/severity — no GPU or model weights required to run `pytest`.
- **Packaging**: `Dockerfile`, `docker-compose.yml` (app + Prometheus), `pyproject.toml`,
  `requirements.txt`.

## Tier 2 — scaffolded, needs your data/GPU

- `sentinel/actions/` — pretrained Kinetics-400 action recognition (zero-shot) is real and runs;
  `training/train_action_classifier.py` fine-tunes on your own labeled clips once you have them.
- `training/train_anomaly_autoencoder.py` — LSTM autoencoder over trajectories; trains on
  whatever "normal" trajectory data you record with this system.
- `scripts/export_onnx.py` — real ONNX export for the YOLO model.

## Tier 3 — roadmap only

See [ROADMAP.md](ROADMAP.md): TensorRT/Jetson edge deployment, MLflow+DVC server setup,
Grafana dashboards, formal MOT17/COCO benchmark runs, the 8-hour soak test, demo video, blog post.

## Success criteria (revised, honest version)

| Metric | Target | How to check |
|---|---|---|
| End-to-end run | 3 concurrent sources (webcam + 2 video files, or RTSP) process without crashing | `python -m sentinel.main --config config/cameras.yaml` |
| Alert correctness | Staged test clips (a person sitting down fast, a bag left behind, a crowd of >N people) each produce the correct alert type | `tests/test_anomaly_scenarios.py` + manual run on `data/raw/sample_clips/` |
| Dedup | Repeated triggers of the same event on the same track within the cooldown window produce exactly one alert | `tests/test_alert_manager.py` |
| FPS | Measured and reported for your actual hardware (not assumed) | `python scripts/benchmark_fps.py` |
| Unit tests | All pass, no GPU required | `pytest` |
| Formal MOTA/mAP numbers | Documented as a follow-up once you run the official eval harness against MOT17/COCO | [ROADMAP.md](ROADMAP.md) |
