"""
Sentinex AI - Hyperparameter tuning.

Uses Ultralytics model.tune() for automated optimization.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def tune(
    data_yaml: str = "datasets/open_images_weapons/data.yaml",
    model_size: str = "x",
    epochs: int = 30,
    iterations: int = 100,
):
    data_path = Path(data_yaml)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_path}")

    model = YOLO(f"yolov8{model_size}.pt")

    print("=" * 60)
    print("SENTINEX AI - HYPERPARAMETER TUNING")
    print(f"Iterations: {iterations} x {epochs} epochs")
    print("=" * 60)

    model.tune(
        data=str(data_path),
        epochs=epochs,
        iterations=iterations,
        optimizer="AdamW",
        plots=True,
        save=False,
        val=True,
    )

    print("\nTuning complete. Check tune_results.csv for best hyperparameters.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets/open_images_weapons/data.yaml")
    parser.add_argument("--model", default="x", choices=["n", "s", "m", "l", "x"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    tune(args.data, args.model, args.epochs, args.iterations)
