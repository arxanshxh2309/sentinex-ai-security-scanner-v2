# Sentinex AI Security Scanner v2

AI-assisted baggage and X-ray threat detection using YOLOv8, FastAPI, and
Bayesian threat scoring.

> This project is a research/prototype scanner. It is not a certified security
> screening system and should not be used as the sole basis for safety decisions.

## Features

- YOLOv8 training pipeline for weapon and prohibited-item detection.
- Google Open Images V7 downloader for Knife, Handgun, Dagger, Scissors, and Sword.
- Optional X-ray preprocessing with CLAHE contrast enhancement.
- FastAPI service with `POST /scan`.
- Bayesian threat assessment with a fast production path and optional PyMC sampling.
- Export support for ONNX, TensorRT, and TorchScript.
- Google Colab notebook for GPU training.

## Repository Layout

```text
Sentinex_Max_Training.ipynb     Google Colab notebook
api/main.py                     FastAPI scanner service
datasets/download_datasets.py   Open Images downloader
datasets/preprocess.py          X-ray preprocessing
datasets/process_xray_dataset.py Dataset splitting
mcmc/bayesian_model.py          Threat probability estimation
training/train_yolo.py          Main YOLO training script
training/evaluate.py            Evaluation and benchmarks
training/export_model.py        Model export
training/tune_hyperparams.py    Hyperparameter tuning
```

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Download a training dataset:

```bash
python datasets/download_datasets.py --source openimages
```

Train:

```bash
python training/train_yolo.py --model x --epochs 300
```

Evaluate:

```bash
python training/evaluate.py
```

Export:

```bash
python training/export_model.py
```

## Run the API

By default the API looks for:

```text
runs/sentinex/max_train/weights/best.pt
```

Start the server:

```bash
uvicorn api.main:app --reload
```

Scan an image:

```bash
curl -X POST http://localhost:8000/scan -F "file=@your_image.jpg"
```

Useful environment variables are documented in `.env.example`:

```bash
SENTINEX_MODEL=runs/sentinex/max_train/weights/best.pt
SENTINEX_CONF=0.15
SENTINEX_IOU=0.45
SENTINEX_IMGSZ=640
SENTINEX_TTA=false
SENTINEX_MCMC_SAMPLES=0
```

`SENTINEX_CONF=0.15` favors recall, which is usually better for screening. Raise
it if false positives become too noisy. Enable `SENTINEX_TTA=true` for slower
but often more sensitive inference.

## Docker

```bash
docker build -t sentinex-ai-security-scanner .
docker run --rm -p 8000:8000 -e SENTINEX_MODEL=/models/best.pt -v "${PWD}/runs/sentinex/max_train/weights:/models" sentinex-ai-security-scanner
```

## Tests

```bash
python -m unittest discover -s tests
```

## Improving Detection Success Rate

For higher recall and better real-world performance:

- Train on real baggage/X-ray imagery, not only natural Open Images photos.
- Add hard negatives: keys, tools, electronics, cables, pens, and folded metal objects.
- Keep a separate test set from different scanners or image sources.
- Track per-class precision, recall, mAP50, and confusion matrix after every run.
- Start with lower inference confidence (`0.10` to `0.20`) for screening workflows.
- Use test-time augmentation for high-risk scans when latency is acceptable.
- Review false negatives first, then add targeted examples and retrain.

## GitHub Notes

Large generated artifacts are intentionally ignored:

- downloaded datasets
- `runs/`
- trained weights (`*.pt`, `*.onnx`, `*.engine`, `*.torchscript`)
- Python caches

Upload model weights separately through a release, cloud bucket, or model registry
instead of committing them directly.


## License

This project's code is licensed under the MIT License — see the [LICENSE](LICENSE.md) file for details.

Note: this project depends on Ultralytics YOLOv8, which is licensed under **AGPL-3.0**. That dependency carries its own copyleft terms separate from this repository's MIT license. If you use or distribute this project with YOLOv8, review Ultralytics' license obligations.

