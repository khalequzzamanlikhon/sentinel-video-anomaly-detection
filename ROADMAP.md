# Roadmap (Tier 3) — not attempted in this repo, here's exactly how to do each one

These need real hardware, long-running environments, curated datasets, or content creation
that belong to you, not a single coding session. Each item below is scoped so you can pick it
up directly.

## 1. Formal detection/tracking benchmarks (MOT17, COCO)

1. Download MOT17 (`https://motchallenge.net/data/MOT17/`) and COCO val2017.
2. `pip install motmetrics`
3. Run your tracker over MOT17's `train` sequences (the ones with public ground truth) and
   compute MOTA/IDF1 with `motmetrics.utils.compare_to_groundtruth` — there's no shortcut here;
   the numbers only mean something if computed against the official split.
4. For COCO mAP, use Ultralytics' own `model.val(data="coco.yaml")` — it's built in.
5. Drop the resulting numbers into your README's Success Criteria table, with the exact command
   used, so it's reproducible by a reader.

## 2. Custom action classifier fine-tuning

1. Collect ~20+ clips per class (walking, running, falling, fighting, loitering). Options:
   - Record yourself/friends performing them.
   - Pull matching Kinetics-400 clips (`falling off bike`, `wrestling`, `walking the dog`, etc.
     are close analogues) — check the license before publishing any of the source clips.
   - Use MediaPipe pose estimation to synthesize fall sequences by manipulating joint angles
     (the spec's Challenge 5 suggestion) if you want more fall examples than you can film.
2. Label with Label Studio (`pip install label-studio`) — even a spreadsheet mapping
   filename → class works for this scale.
3. Arrange as `data/annotations/actions/<class>/*.mp4` and run
   `training/train_action_classifier.py` (already in this repo, genuinely runs).
4. Evaluate on a held-out 20% split; report per-class Top-1 accuracy.

## 3. LSTM autoencoder on real "normal" trajectories

1. Add trajectory export to `TrajectoryStore` (dump each completed track's `points` on
   eviction) or just log them from `SentinelPipeline.step()`.
2. Record several hours of ordinary foot traffic through the camera(s) you'll actually deploy on.
3. Run `training/train_anomaly_autoencoder.py` (already in this repo) on the exported trajectories.
4. Pick a reconstruction-error threshold from the trained model's error distribution on a
   held-out normal set (e.g., 99th percentile), and wire it into `RuleEngine` as an additional
   signal alongside the rule-based detectors.

## 4. TensorRT + edge deployment (Jetson Nano / Raspberry Pi 5 + Coral)

1. `python scripts/export_onnx.py --weights yolov8n.pt` (already in this repo, genuinely runs).
2. On the target device (needs an actual NVIDIA Jetson for TensorRT):
   `trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n.engine --fp16`
3. Swap `YoloEngine` to load the `.engine` file via Ultralytics' TensorRT backend
   (`YOLO("yolov8n.engine")` — Ultralytics detects the format from the extension).
4. For Raspberry Pi 5 + Coral TPU instead: export to TFLite (`model.export(format="tflite")`)
   and use `pycoral` for the Edge TPU runtime; drop BoT-SORT for ByteTrack (`tracker.tracker:
   bytetrack.yaml`) since ReID appearance matching is too slow on that hardware; run action
   recognition on a server instead of on-device.
5. Re-run `scripts/benchmark_fps.py` on the device and record the real FPS.

## 5. MLflow + DVC servers

- MLflow: `pip install mlflow`, then `mlflow ui` locally, or point `MLFLOW_TRACKING_URI` at a
  shared server. Both training scripts in `training/` already call `mlflow.log_metric` /
  `log_params` when run with `--mlflow` — no code changes needed, just start the server.
- DVC: `pip install dvc`, `dvc init`, `dvc add data/raw/<dataset>`, then `dvc remote add` pointing
  at S3/GCS/a shared drive, and `dvc push`. Do this once you've actually downloaded a dataset in
  step 1 above — there's nothing to version yet otherwise.

## 6. Grafana dashboards on top of Prometheus

The Prometheus metrics endpoint in this repo (`sentinel/core/metrics.py`, port 8000) is real
and already exports `inference_fps`, `alert_count`, `stream_lag_seconds`. To visualize:
1. `docker compose -f docker/docker-compose.yml up prometheus`
2. Run Grafana (`docker run -p 3000:3000 grafana/grafana`), add Prometheus
   (`http://prometheus:9090`) as a data source, and build panels for the three metrics above.

## 7. 8-hour soak test

`python scripts/stress_test.py --duration 8h --config config` (already in this repo, genuinely
runs) — needs 8 real hours of wall-clock time against real camera feeds. Report the printed
summary (crashes, memory growth, uptime %) in your README once you've run it.

## 8. Demo video + blog post

Once the above produce real numbers and a working demo:
- Record the 4 scenes from the original spec (3-camera tracking, a staged fall → alert, an
  abandoned bag over time, the dashboard analytics view).
- Write the blog post using your *actual* results — the interesting story is usually where the
  rule-based approach beat/lost to the learned one, and what the real measured FPS was, not the
  planned numbers.
