"""Fine-tunes the action-recognition backbone (sentinel/actions/slowfast_engine.py) on your
own labeled clips, replacing the pretrained Kinetics-400 head with one for your target classes
(default: walking, running, falling, fighting, loitering -- matching the original spec).

This needs labeled clips first. The spec's suggested sources: record ~20 clips per class
yourself, augment with matching Kinetics-400/UCF-101 clips, label with Label Studio. See
ROADMAP.md for the full data-collection plan; this script assumes you already have:

    data/annotations/actions/
        walking/clip_001.mp4, clip_002.mp4, ...
        running/...
        falling/...
        fighting/...
        loitering/...

Usage:
    python training/train_action_classifier.py --data data/annotations/actions \
        --backend x3d --epochs 10 --out sentinel_action_classifier.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_clip_frames(path: str, num_frames: int = 13, size: int = 182) -> np.ndarray:
    import cv2

    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(cv2.resize(frame, (size, size)), cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"No frames read from {path}")
    idx = np.linspace(0, len(frames) - 1, num_frames).astype(int)
    return np.stack([frames[i] for i in idx])


def build_dataset(data_dir: str) -> tuple[list[str], list[int], list[str]]:
    classes = sorted(p.name for p in Path(data_dir).iterdir() if p.is_dir())
    paths, labels = [], []
    for label_idx, cls in enumerate(classes):
        for clip_path in (Path(data_dir) / cls).glob("*.mp4"):
            paths.append(str(clip_path))
            labels.append(label_idx)
    if not paths:
        raise SystemExit(
            f"No .mp4 clips found under {data_dir}/<class>/. See this script's docstring for "
            "the expected directory layout."
        )
    return paths, labels, classes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--backend", choices=["x3d", "slowfast"], default="x3d")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", default="sentinel_action_classifier.pt")
    parser.add_argument("--mlflow", action="store_true")
    args = parser.parse_args()

    import torch

    from sentinel.actions.slowfast_engine import SlowFastEngine

    paths, labels, classes = build_dataset(args.data)
    print(f"Found {len(paths)} clips across {len(classes)} classes: {classes}")

    engine = SlowFastEngine(backend=args.backend, device="cpu")

    # Replace the pretrained 400-way head with one sized for your classes; freeze the backbone
    # so a handful of clips per class is enough to get a reasonable starting point.
    in_features = engine.model.blocks[-1].proj.in_features
    engine.model.blocks[-1].proj = torch.nn.Linear(in_features, len(classes))
    for name, param in engine.model.named_parameters():
        param.requires_grad = "blocks.5" in name or "blocks.6" in name  # last stage + head only

    optimizer = torch.optim.Adam(
        (p for p in engine.model.parameters() if p.requires_grad), lr=args.lr
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    mlflow = None
    if args.mlflow:
        try:
            import mlflow as mlflow_module

            mlflow = mlflow_module
            mlflow.start_run()
            mlflow.log_params(vars(args))
        except ImportError:
            print("--mlflow passed but mlflow isn't installed (pip install -e '.[mlops]'); skipping logging")

    engine.model.train()
    for epoch in range(args.epochs):
        total_loss, correct = 0.0, 0
        for path, label in zip(paths, labels):
            clip = load_clip_frames(path, num_frames=engine.num_frames, size=engine.crop_size)
            inputs = engine._preprocess(clip)  # noqa: SLF001 -- reuse the exact inference preprocessing
            target = torch.tensor([label])

            optimizer.zero_grad()
            logits = engine.model(inputs)
            loss = loss_fn(logits, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += int(logits.argmax(dim=1).item() == label)

        acc = correct / len(paths)
        print(f"epoch {epoch + 1}/{args.epochs} - loss {total_loss / len(paths):.4f} - train acc {acc:.3f}")
        if mlflow:
            mlflow.log_metrics({"loss": total_loss / len(paths), "train_acc": acc}, step=epoch)

    torch.save(engine.model.state_dict(), args.out)
    Path(args.out + ".classes.txt").write_text("\n".join(classes))
    print(f"Saved fine-tuned weights to {args.out} (classes in {args.out}.classes.txt)")
    if mlflow:
        mlflow.log_artifact(args.out)
        mlflow.end_run()


if __name__ == "__main__":
    main()
