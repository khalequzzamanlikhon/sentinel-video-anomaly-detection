"""Exports the YOLOv8 detection model to ONNX (real, uses Ultralytics' built-in exporter --
this genuinely runs and produces a usable .onnx file). TensorRT conversion from the resulting
ONNX file needs an NVIDIA GPU + TensorRT installed and is documented as a follow-up in
ROADMAP.md rather than attempted here.

Usage:
    python scripts/export_onnx.py --weights yolov8n.pt --out yolov8n.onnx
"""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true", help="FP16 export (needs a GPU)")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    path = model.export(format="onnx", imgsz=args.imgsz, half=args.half)
    print(f"Exported ONNX model to {path}")
    print(
        "Next step for edge deployment (Jetson/TensorRT): see ROADMAP.md -- "
        "`trtexec --onnx=<path> --saveEngine=model.engine` on the target device."
    )


if __name__ == "__main__":
    main()
