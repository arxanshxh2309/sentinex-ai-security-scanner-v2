"""
Sentinex AI - Comprehensive model evaluation.

Generates mAP, precision, recall, TTA metrics, and speed benchmarks.
"""

import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO


def evaluate(
    weights: str = "runs/sentinex/max_train/weights/best.pt",
    data_yaml: str = "datasets/open_images_weapons/data.yaml",
    imgsz: int = 640,
    split: str = "test",
):
    print("=" * 60)
    print("SENTINEX AI - FULL EVALUATION")
    print("=" * 60)

    weights_path = Path(weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    if not Path(data_yaml).exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_yaml}")

    model = YOLO(str(weights_path))

    metrics = model.val(
        data=data_yaml,
        imgsz=imgsz,
        batch=8,
        split=split,
        plots=True,
        save_json=True,
    )

    results = {
        "model": str(weights_path),
        "split": split,
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }

    print("\nRunning TTA evaluation...")
    tta_metrics = model.val(
        data=data_yaml,
        imgsz=imgsz,
        batch=4,
        split=split,
        augment=True,
    )
    results["tta_mAP50"] = float(tta_metrics.box.map50)
    results["tta_mAP50_95"] = float(tta_metrics.box.map)

    print("\nRunning speed benchmark...")
    try:
        speed = model.benchmark(
            data=data_yaml,
            imgsz=imgsz,
            half=torch.cuda.is_available(),
        )
        results["benchmark"] = speed
    except Exception as exc:
        results["benchmark_error"] = str(exc)
        print(f"Benchmark skipped: {exc}")

    out_path = weights_path.parent.parent / "evaluation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"mAP@50:      {results['mAP50']:.4f}")
    print(f"mAP@50-95:   {results['mAP50_95']:.4f}")
    print(f"Precision:   {results['precision']:.4f}")
    print(f"Recall:      {results['recall']:.4f}")
    print(f"TTA mAP@50:  {results['tta_mAP50']:.4f}")
    print(f"\nSaved to: {out_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="runs/sentinex/max_train/weights/best.pt")
    parser.add_argument("--data", default="datasets/open_images_weapons/data.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()
    evaluate(args.weights, args.data, args.imgsz, args.split)
