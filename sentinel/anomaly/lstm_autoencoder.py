"""LSTM autoencoder over trajectories (Tier 2, see REVISED_PLAN.md).

Trained on "normal" trajectories only; a track whose reconstruction error exceeds `threshold`
is flagged as an unknown-pattern anomaly, catching cases the rule engine doesn't have a
specific rule for. Training is done by `training/train_anomaly_autoencoder.py` once you've
recorded normal-behaviour footage with this system -- this module defines the model and the
scoring path used by both training and inference, so they can't drift apart.

Requires `torch` (already a transitive dependency via `ultralytics`), imported lazily so the
Tier 1 rule-based system never needs it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SEQ_LEN = 30  # ~2 seconds at 15 FPS
FEATURE_DIM = 4  # (dx, dy, w, h) per frame, normalized


class TrajectoryAutoencoder:
    """Thin wrapper so callers don't need to know the torch module layout."""

    def __init__(self, hidden_dim: int = 32, num_layers: int = 1) -> None:
        import torch
        import torch.nn as nn

        self._torch = torch

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = nn.LSTM(FEATURE_DIM, hidden_dim, num_layers, batch_first=True)
                self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
                self.output = nn.Linear(hidden_dim, FEATURE_DIM)

            def forward(self, x):
                _, (h, c) = self.encoder(x)
                latent = h[-1].unsqueeze(1).repeat(1, x.size(1), 1)
                decoded, _ = self.decoder(latent)
                return self.output(decoded)

        self.model = _Model()

    def load(self, weights_path: str | Path) -> None:
        state = self._torch.load(weights_path, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()

    def save(self, weights_path: str | Path) -> None:
        self._torch.save(self.model.state_dict(), weights_path)

    def reconstruction_error(self, sequence: np.ndarray) -> float:
        """`sequence` is (SEQ_LEN, FEATURE_DIM) normalized trajectory features."""
        with self._torch.no_grad():
            x = self._torch.tensor(sequence, dtype=self._torch.float32).unsqueeze(0)
            recon = self.model(x)
            error = self._torch.mean((recon - x) ** 2).item()
        return error


def trajectory_to_features(points: list, seq_len: int = SEQ_LEN) -> np.ndarray | None:
    """Convert a Track's point history into a fixed-length (dx, dy, w, h) feature sequence,
    normalized by the first point's scale. Returns None if there isn't enough history yet.
    """
    if len(points) < seq_len:
        return None
    window = points[-seq_len:]
    x0, y0 = window[0].x, window[0].y
    scale = max(window[0].w, 1.0)
    feats = np.array(
        [[(p.x - x0) / scale, (p.y - y0) / scale, p.w / scale, p.h / scale] for p in window],
        dtype=np.float32,
    )
    return feats
