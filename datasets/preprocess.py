"""
Sentinex AI — X-ray Image Preprocessing
CLAHE contrast enhancement, resize, and noise reduction for X-ray images.
"""

import cv2
from pathlib import Path
import argparse


def preprocess_images(
    input_dir: str = "datasets/raw/images",
    output_dir: str = "datasets/processed/preprocessed",
    img_size: int = 640,
):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"Input directory not found: {input_path}")
        return

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    count = 0

    for image_path in input_path.glob("*"):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Skipping unreadable: {image_path.name}")
            continue

        # CLAHE contrast enhancement (better than equalizeHist for X-ray)
        image = clahe.apply(image)
        # Denoise
        image = cv2.fastNlMeansDenoising(image, h=10)
        # Resize
        image = cv2.resize(image, (img_size, img_size))

        cv2.imwrite(str(output_path / image_path.name), image)
        count += 1

    print(f"Preprocessed {count} images -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentinex AI — X-ray Preprocessing")
    parser.add_argument("--input", default="datasets/raw/images")
    parser.add_argument("--output", default="datasets/processed/preprocessed")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    preprocess_images(args.input, args.output, args.imgsz)
