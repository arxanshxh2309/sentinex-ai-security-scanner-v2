"""
Sentinex AI Security Scanner - YOLO training script.

For Google Colab, use Sentinex_Max_Training.ipynb. For local training:
python training/train_yolo.py
"""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def train(
    model_size: str = "x",
    data_yaml: str = "datasets/open_images_weapons/data.yaml",
    epochs: int = 300,
    imgsz: int = 640,
    device: str = "auto",
    batch: int | None = None,
    val_split: str = "test",
):
    """Train YOLOv8 with recall-focused settings for screening."""
    data_path = Path(data_yaml)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_path}")

    if device == "auto":
        device = 0 if torch.cuda.is_available() else "cpu"

    if batch is None:
        batch = -1 if torch.cuda.is_available() else 8

    model_name = f"yolov8{model_size}.pt"
    print(f"\n{'=' * 60}")
    print("SENTINEX AI - TRAINING")
    print(f"{'=' * 60}")
    print(f"Model:      {model_name}")
    print(f"Dataset:    {data_path}")
    print(f"Epochs:     {epochs}")
    print(f"Image size: {imgsz}")
    print(f"Device:     {device}")
    print(f"Batch:      {batch}")
    print(f"{'=' * 60}\n")

    model = YOLO(model_name)

    results = model.train(
        data=str(data_path),
        epochs=epochs,
        patience=50,
        imgsz=imgsz,
        batch=batch,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.3,
        degrees=15.0,
        translate=0.2,
        scale=0.9,
        shear=5.0,
        perspective=0.001,
        flipud=0.5,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.4,
        erasing=0.3,
        close_mosaic=15,
        label_smoothing=0.1,
        project="runs/sentinex",
        name="max_train",
        save=True,
        save_period=10,
        plots=True,
        device=device,
        workers=4,
        verbose=True,
        seed=42,
        deterministic=True,
        exist_ok=True,
    )

    print(f"\n{'=' * 60}")
    print("EVALUATING BEST MODEL")
    print(f"{'=' * 60}\n")

    best_path = Path("runs/sentinex/max_train/weights/best.pt")
    best_model = YOLO(str(best_path))
    metrics = best_model.val(data=str(data_path), imgsz=imgsz, split=val_split, plots=True)

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print(f"mAP@50:      {metrics.box.map50:.4f}")
    print(f"mAP@50-95:   {metrics.box.map:.4f}")
    print(f"Precision:   {metrics.box.mp:.4f}")
    print(f"Recall:      {metrics.box.mr:.4f}")
    print(f"{'=' * 60}")
    print(f"\nBest weights: {best_path.resolve()}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentinex AI YOLO training")
    parser.add_argument("--model", default="x", choices=["n", "s", "m", "l", "x"])
    parser.add_argument("--data", default="datasets/open_images_weapons/data.yaml")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--val-split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    train(
        model_size=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        batch=args.batch,
        val_split=args.val_split,
    )
