"""
Sentinex AI — Dataset Splitting
Splits raw images + labels into train/val/test with stratified splitting.
"""

from pathlib import Path
import shutil
import argparse
import random


def process_dataset(
    raw_images: str = "datasets/raw/images",
    raw_labels: str = "datasets/raw/labels",
    output_dir: str = "datasets/processed",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    raw_img_path = Path(raw_images)
    raw_lbl_path = Path(raw_labels)
    output_path = Path(output_dir)

    if not raw_img_path.exists():
        print(f"Image directory not found: {raw_img_path}")
        return

    # Collect images
    images = sorted(
        [p for p in raw_img_path.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    )

    if not images:
        print("No images found!")
        return

    # Shuffle deterministically
    random.seed(seed)
    random.shuffle(images)

    # Split
    n = len(images)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:],
    }

    for split_name, split_images in splits.items():
        img_dir = output_path / split_name / "images"
        lbl_dir = output_path / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img in split_images:
            shutil.copy(img, img_dir / img.name)

            # Copy corresponding label file if it exists
            label_file = raw_lbl_path / f"{img.stem}.txt"
            if label_file.exists():
                shutil.copy(label_file, lbl_dir / label_file.name)

    print("=" * 50)
    print("DATASET SPLIT COMPLETE")
    print("=" * 50)
    for split_name, split_images in splits.items():
        print(f"  {split_name:6s}: {len(split_images)} images")
    print(f"\nOutput: {output_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentinex AI — Dataset Splitting")
    parser.add_argument("--images", default="datasets/raw/images")
    parser.add_argument("--labels", default="datasets/raw/labels")
    parser.add_argument("--output", default="datasets/processed")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()
    process_dataset(args.images, args.labels, args.output, args.train_ratio, args.val_ratio)
