"""
Sentinex AI - Model export.

Exports a trained YOLO model to deployment formats.
"""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def export_model(weights: str = "runs/sentinex/max_train/weights/best.pt", imgsz: int = 640):
    print("=" * 60)
    print("SENTINEX AI - MODEL EXPORT")
    print("=" * 60)

    weights_path = Path(weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    model = YOLO(str(weights_path))
    use_half = torch.cuda.is_available()

    print("\n[1/3] Exporting ONNX...")
    model.export(format="onnx", imgsz=imgsz, half=use_half, simplify=True)
    print("ONNX exported")

    print("\n[2/3] Exporting TensorRT...")
    try:
        model.export(format="engine", imgsz=imgsz, half=use_half)
        print("TensorRT exported")
    except Exception as exc:
        print(f"TensorRT skipped: {exc}")

    print("\n[3/3] Exporting TorchScript...")
    model.export(format="torchscript", imgsz=imgsz)
    print("TorchScript exported")

    print(f"\n{'=' * 60}")
    print("EXPORT COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="runs/sentinex/max_train/weights/best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    export_model(args.weights, args.imgsz)
