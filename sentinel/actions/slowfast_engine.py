"""Pretrained SlowFast / X3D action recognition via PyTorchVideo (Tier 2, see REVISED_PLAN.md).

Ships with zero-shot Kinetics-400 labels out of the box -- no training needed to get real
predictions ("walking", "running", "falling on floor", ... are Kinetics-400 classes). Fine-
tuning on the 5 custom classes from the original spec (`training/train_action_classifier.py`)
replaces the final classification head once you have labeled clips.

Requires the `actions` extra: `pip install -e ".[actions]"` (pytorchvideo + a torch.hub
download on first use, ~130MB for slowfast_r50). Lazy-imported so Tier 1 never pays this cost.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

KINETICS_400_URL = (
    "https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json"
)


@dataclass
class ActionPrediction:
    label: str
    confidence: float


class SlowFastEngine:
    """backend='slowfast' uses slowfast_r50 (32-frame clips, slow+fast pathways).
    backend='x3d' uses x3d_s (13-frame clips, single pathway, faster/lighter -- recommended
    for the short/sparse clips described in Challenge 4).
    """

    def __init__(self, backend: str = "x3d", device: str = "cpu", weights_path: str | None = None) -> None:
        import torch

        self._torch = torch
        self.backend = backend
        self.device = device
        self.class_names = self._load_class_names()

        if backend == "x3d":
            self.model = torch.hub.load("facebookresearch/pytorchvideo", "x3d_s", pretrained=True)
            self.num_frames, self.sampling_rate, self.crop_size = 13, 6, 182
        elif backend == "slowfast":
            self.model = torch.hub.load("facebookresearch/pytorchvideo", "slowfast_r50", pretrained=True)
            self.num_frames, self.sampling_rate, self.crop_size = 32, 2, 256
        else:
            raise ValueError(f"Unknown backend {backend!r}")

        if weights_path:
            state = torch.load(weights_path, map_location=device)
            self.model.load_state_dict(state)
            self.class_names = None  # fine-tuned head uses your own label list, set separately

        self.model = self.model.eval().to(device)

    def _load_class_names(self) -> dict[int, str] | None:
        import json
        import urllib.request

        try:
            with urllib.request.urlopen(KINETICS_400_URL, timeout=5) as resp:
                name_to_id = json.loads(resp.read())
            return {v: k.replace('"', "") for k, v in name_to_id.items()}
        except Exception:
            return None  # offline: predictions will be returned as raw class indices

    def _preprocess(self, clip: np.ndarray):
        """clip: (T, H, W, 3) uint8 RGB, T >= 8 (Challenge 4: short clips are fine -- we
        subsample/pad to the model's required length rather than requiring a fixed 2s)."""
        torch = self._torch
        import torch.nn.functional as F

        t = torch.from_numpy(clip).permute(3, 0, 1, 2).float() / 255.0  # (C, T, H, W)
        mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
        std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)
        t = (t - mean) / std

        # temporal sample/pad to num_frames
        T = t.shape[1]
        idx = torch.linspace(0, T - 1, self.num_frames).long().clamp(max=T - 1)
        t = t[:, idx, :, :]

        t = F.interpolate(t.unsqueeze(0), size=(self.num_frames, self.crop_size, self.crop_size), mode="trilinear", align_corners=False)

        if self.backend == "slowfast":
            fast = t
            slow_idx = torch.linspace(0, self.num_frames - 1, self.num_frames // 4).long()
            slow = t[:, :, slow_idx, :, :]
            return [slow.to(self.device), fast.to(self.device)]
        return t.to(self.device)

    def predict(self, clip: np.ndarray, top_k: int = 1) -> list[ActionPrediction]:
        torch = self._torch
        inputs = self._preprocess(clip)
        with torch.no_grad():
            logits = self.model(inputs)
            probs = torch.nn.functional.softmax(logits, dim=1)[0]
        top = torch.topk(probs, k=min(top_k, probs.shape[0]))
        out = []
        for score, idx in zip(top.values.tolist(), top.indices.tolist()):
            label = self.class_names.get(idx, str(idx)) if self.class_names else str(idx)
            out.append(ActionPrediction(label=label, confidence=score))
        return out
