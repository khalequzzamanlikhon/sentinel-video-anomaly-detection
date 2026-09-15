"""Trains the LSTM trajectory autoencoder (sentinel/anomaly/lstm_autoencoder.py) on "normal"
trajectories -- i.e., trajectories recorded during ordinary operation with no anomalies.

This is a genuine, runnable training loop; what it does NOT do is invent training data for
you. Collect normal-trajectory data first:

    python -m sentinel.main --config config --headless
    # let it run against real footage of ordinary activity for a while; every track's
    # (x, y, w, h, t) history is exactly the input this script needs. Wire up trajectory
    # export in TrajectoryStore (dump each completed track's `points` to a .npy/.jsonl file)
    # before running this, or adapt `load_trajectories` below to your export format.

Usage:
    python training/train_anomaly_autoencoder.py --data data/processed/normal_trajectories/ \
        --epochs 20 --out sentinel_autoencoder.pt
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from sentinel.anomaly.lstm_autoencoder import FEATURE_DIM, SEQ_LEN, TrajectoryAutoencoder


def load_trajectories(data_dir: str) -> np.ndarray:
    """Expects one JSON file per track: {"points": [{"x":.., "y":.., "w":.., "h":.., "timestamp":..}, ...]}.
    Produces (N, SEQ_LEN, FEATURE_DIM) sliding-window sequences.
    """
    sequences = []
    for path in glob.glob(str(Path(data_dir) / "*.json")):
        with open(path) as f:
            record = json.load(f)
        points = record["points"]
        if len(points) < SEQ_LEN:
            continue
        x0, y0 = points[0]["x"], points[0]["y"]
        scale = max(points[0]["w"], 1.0)
        feats = np.array(
            [[(p["x"] - x0) / scale, (p["y"] - y0) / scale, p["w"] / scale, p["h"] / scale] for p in points],
            dtype=np.float32,
        )
        for start in range(0, len(feats) - SEQ_LEN + 1, SEQ_LEN // 2):
            sequences.append(feats[start : start + SEQ_LEN])
    if not sequences:
        raise SystemExit(
            f"No trajectory sequences of length >= {SEQ_LEN} found in {data_dir}. "
            "See this script's module docstring for the expected input format."
        )
    return np.stack(sequences)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Directory of per-track JSON trajectory files")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="sentinel_autoencoder.pt")
    parser.add_argument("--mlflow", action="store_true", help="Log metrics to MLflow if installed")
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    sequences = load_trajectories(args.data)
    print(f"Loaded {len(sequences)} training sequences of shape (seq_len={SEQ_LEN}, dim={FEATURE_DIM})")

    dataset = TensorDataset(torch.tensor(sequences))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    wrapper = TrajectoryAutoencoder()
    optimizer = torch.optim.Adam(wrapper.model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    mlflow = None
    if args.mlflow:
        try:
            import mlflow as mlflow_module

            mlflow = mlflow_module
            mlflow.start_run()
            mlflow.log_params(vars(args))
        except ImportError:
            print("--mlflow passed but mlflow isn't installed (pip install -e '.[mlops]'); skipping logging")

    for epoch in range(args.epochs):
        total_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            recon = wrapper.model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"epoch {epoch + 1}/{args.epochs} - reconstruction MSE: {avg_loss:.6f}")
        if mlflow:
            mlflow.log_metric("reconstruction_mse", avg_loss, step=epoch)

    wrapper.save(args.out)
    print(f"Saved trained autoencoder weights to {args.out}")
    if mlflow:
        mlflow.log_artifact(args.out)
        mlflow.end_run()


if __name__ == "__main__":
    main()
