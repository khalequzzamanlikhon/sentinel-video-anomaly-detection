# Setup, Run & Git Guide

## 1. Prerequisites

- Python 3.10+ (developed/tested on 3.12)
- Windows/macOS/Linux — OpenCV and Ultralytics all run cross-platform
- A GPU is **not required**. It runs on CPU (slower); if you have an NVIDIA GPU with CUDA set
  up, Ultralytics will use it automatically (`device: auto` in `config/models.yaml`).

### Important: use a fresh virtual environment, not your base/global Python

While building this repo, `pip install ultralytics` into the ambient environment upgraded
`torch` (2.7.0 → 2.14.0) and `httpx`, which produced a dependency-conflict warning against an
unrelated `ollama` package already installed there. Nothing broke, but it's exactly the kind of
cross-project interference a dedicated virtual environment avoids. Always create one for this
project:

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Windows (Git Bash / WSL / macOS / Linux)
source .venv/bin/activate
```

## 2. Install

```bash
pip install --upgrade pip
pip install -e .
```

This installs OpenCV, NumPy, Ultralytics (YOLOv8 + BoT-SORT/ByteTrack), PyYAML, Streamlit,
pandas, and prometheus-client — everything Tier 1 needs. Tier 2 extras (action recognition,
MLOps) are optional:

```bash
pip install -e ".[actions]"   # SlowFast/X3D action recognition (torch, torchvision, pytorchvideo)
pip install -e ".[mlops]"     # MLflow + DVC for the training scripts
pip install -e ".[dev]"       # pytest, ruff (pytest is already in the base deps too)
```

## 3. Generate demo clips (optional but recommended for a first run)

```bash
python scripts/make_sample_clips.py
```

Writes two small synthetic video files to `data/raw/sample_clips/`, which is what
`config/cameras.yaml` points at by default. These are only useful for an I/O smoke test
(capture → tracking IDs → dashboard) — they're moving rectangles, not people, so YOLO won't
detect anything interesting in them. For a real demo, either:

- Point a `cameras.yaml` entry at your webcam (`type: webcam`, `source: 0`), or
- Point it at a real video file with people in it (`type: file`, `source: path/to/clip.mp4`).

## 4. Run the pipeline (CLI, live OpenCV windows)

```bash
python -m sentinel.main --config config
```

- One OpenCV window per configured source, with bounding boxes, track IDs, and alert banners.
- Press `q` in any window to quit.
- `--headless` runs without opening windows (useful over SSH or for `scripts/stress_test.py`).
- `--metrics-port 8000` (default) exposes Prometheus metrics at `http://localhost:8000/metrics`.

First run downloads `yolov8n.pt` (~6MB) automatically via Ultralytics — needs internet access
once; the file is then cached locally (and gitignored — see `.gitignore`).

## 5. Run the dashboard

```bash
streamlit run sentinel/dashboard/app.py
```

Opens at `http://localhost:8501`. The **Home** page shows live feeds; **Alerts** (sidebar)
shows history with thumbnails, clip playback, false-positive flagging, and CSV export;
**Analytics** shows charts. The dashboard runs its own copy of the pipeline (don't run
`sentinel.main` and the dashboard against the same webcam device simultaneously — most cameras
only allow one reader at a time).

## 6. Run the tests

```bash
pytest
```

Should show `17 passed` — pure logic, no GPU or model weights needed. Run this after any change
to `sentinel/anomaly/`, `sentinel/tracking/trajectory_store.py`, or `sentinel/alerts/`.

## 7. Benchmark & stress test (produces real numbers for your README)

```bash
python scripts/benchmark_fps.py --source data/raw/sample_clips/sample1.mp4 --frames 200
python scripts/stress_test.py --duration 10m --config config
```

Put the actual printed FPS/uptime numbers into `README.md`'s "Honest performance numbers"
section — don't leave the placeholder text there for a repo you're sharing.

## 8. Docker (optional)

```bash
docker compose -f docker/docker-compose.yml up --build
```

Runs the headless pipeline, the Streamlit dashboard (`localhost:8501`), and Prometheus
(`localhost:9090`) as three services, with `data/` and `config/` mounted from the host so
alerts/clips persist and config changes don't need a rebuild.

## 9. Customizing detection/alert behavior

- `config/cameras.yaml` — add/remove sources, set per-camera expected flow direction (for the
  wrong-way rule) and ROI.
- `config/models.yaml` — swap `yolov8n.pt` for `yolov8s/m/l/x.pt` (more accurate, slower),
  change the tracker (`botsort.yaml` vs `bytetrack.yaml`), adjust `process_every_nth_frame` for
  throughput.
- `config/alerts.yaml` — every rule's thresholds, severity mapping, dedup window, and clip
  length live here. Start here if you're getting false positives/negatives on your footage.

## 10. Git: turning this into a repo and pushing it

This directory is not yet a git repository. To publish it:

```bash
git init
git add .
git status   # sanity check: no yolov8n.pt, no data/alerts.db, no data/raw/<big dataset> staged
git commit -m "Initial commit: Sentinel multi-camera anomaly detection (Tier 1)"
```

The `.gitignore` already excludes model weights (`*.pt`), the runtime alert database and
extracted clips/thumbnails, and any large raw datasets you add later under `data/raw/` (except
the small synthetic `sample_clips/`, which **are** tracked so the repo runs out of the box).
The original planning PDF (`project_sentinel_anomaly_detection.pdf`) is also gitignored by
default since it's a personal planning artifact, not part of the deliverable — remove that line
from `.gitignore` first if you'd rather keep it in the public repo for context.

**Always run `git status` after `git add .` and actually read the file list** before committing
— that's what catches an accidentally-large or accidentally-sensitive file before it's in history.

### Create the GitHub repo and push

Using the GitHub CLI (`gh`, if installed):

```bash
gh repo create sentinel-anomaly-detection --public --source=. --remote=origin --push
```

Or manually:

```bash
git remote add origin https://github.com/<your-username>/sentinel-anomaly-detection.git
git branch -M main
git push -u origin main
```

### Add a license before making it public

```bash
# e.g. MIT — GitHub can generate this for you when creating the repo via the web UI,
# or copy the text from https://opensource.org/license/mit/ into LICENSE
```

### Suggested repo description / topics (for the GitHub page)

- Description: "Multi-camera real-time video anomaly detection — YOLOv8 + BoT-SORT tracking,
  rule-based fall/loitering/abandoned-object/crowd/wrong-way detection, Streamlit dashboard."
- Topics: `computer-vision`, `yolov8`, `object-tracking`, `anomaly-detection`,
  `video-analytics`, `streamlit`, `python`

### Ongoing workflow

```bash
git checkout -b feature/my-change
# ... edit ...
pytest                       # keep the suite green
git add <specific files>     # avoid `git add -A` blindly; review what's staged
git commit -m "Describe the why, not just the what"
git push -u origin feature/my-change
```

Open a PR into `main` (`gh pr create`) rather than pushing straight to `main` once you're
iterating past the initial commit — it gives you a diff to review before it's permanent, and a
natural place to note *why* a rule threshold or config default changed.
