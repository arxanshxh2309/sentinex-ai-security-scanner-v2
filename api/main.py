"""
Sentinex AI Security Scanner - FastAPI backend.

POST /scan uploads an image, runs YOLO detection, and returns a Bayesian
threat assessment.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse


BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

DEFAULT_MODEL_PATH = BASE_DIR / "runs" / "sentinex" / "max_train" / "weights" / "best.pt"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

CONF_THRESHOLD = float(os.getenv("SENTINEX_CONF", "0.15"))
IOU_THRESHOLD = float(os.getenv("SENTINEX_IOU", "0.45"))
IMAGE_SIZE = int(os.getenv("SENTINEX_IMGSZ", "640"))
MCMC_SAMPLES = int(os.getenv("SENTINEX_MCMC_SAMPLES", "0"))
MAX_UPLOAD_MB = int(os.getenv("SENTINEX_MAX_UPLOAD_MB", "15"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
USE_TTA = os.getenv("SENTINEX_TTA", "false").lower() in {"1", "true", "yes", "on"}

app = FastAPI(
    title="Sentinex AI Security Scanner",
    description="AI-powered baggage scanner with YOLOv8 detection and Bayesian threat estimation",
    version="2.1.0",
)

yolo_model = None


def _model_path() -> Path:
    configured = Path(os.getenv("SENTINEX_MODEL", str(DEFAULT_MODEL_PATH)))
    if configured.is_absolute():
        return configured
    return BASE_DIR / configured


def _class_name(names: Any, cls_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(cls_id, cls_id))
    return str(names[cls_id])


def load_model():
    """Load YOLO model on first request."""
    global yolo_model
    if yolo_model is not None:
        return yolo_model

    model_path = _model_path()
    if not model_path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Model file not found: {model_path}. Train a model or set SENTINEX_MODEL.",
        )

    try:
        from ultralytics import YOLO

        yolo_model = YOLO(str(model_path))
        print(f"Model loaded: {model_path}")
        return yolo_model
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model load failed: {exc}") from exc


@app.get("/")
def home():
    return {"status": "Sentinex AI Scanner Online", "version": app.version}


@app.get("/health")
def health():
    model_path = _model_path()
    return {
        "status": "healthy" if model_path.exists() else "model_missing",
        "model_path": str(model_path),
        "model_loaded": yolo_model is not None,
        "inference": {
            "confidence_threshold": CONF_THRESHOLD,
            "iou_threshold": IOU_THRESHOLD,
            "image_size": IMAGE_SIZE,
            "tta": USE_TTA,
            "mcmc_samples": MCMC_SAMPLES,
        },
    }


@app.get("/model/info")
def model_info():
    model = load_model()
    names = model.names
    return {
        "model_path": str(_model_path()),
        "class_names": names,
        "num_classes": len(names),
        "task": model.task,
    }


@app.post("/scan")
async def scan_image(file: UploadFile = File(...)):
    """
    Upload an image for threat scanning.

    Returns YOLO detections plus a threat assessment. The default confidence
    threshold favors recall for security screening; tune SENTINEX_CONF upward
    if false positives become too noisy.
    """
    ext = Path(file.filename or "image.jpg").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(status_code=400, detail=f"Invalid file type: {ext}. Allowed: {allowed}")

    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_MB} MB limit.")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        model = load_model()
        results = model.predict(
            source=tmp_path,
            imgsz=IMAGE_SIZE,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            augment=USE_TTA,
            verbose=False,
        )
        result = results[0]

        detections = []
        image_height, image_width = result.orig_shape[:2]
        image_area = max(float(image_height * image_width), 1.0)

        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bbox_area = max((x2 - x1) * (y2 - y1), 0.0)

            detections.append(
                {
                    "class": _class_name(model.names, cls_id).lower(),
                    "confidence": round(conf, 4),
                    "bbox": [round(x1), round(y1), round(x2), round(y2)],
                    "bbox_area_ratio": round(bbox_area / image_area, 4),
                }
            )

        threat_result = {"threat_level": "CLEAR", "threat_probability": 0.0}
        if detections:
            from mcmc.bayesian_model import estimate_threat

            threat_result = estimate_threat(detections, n_samples=MCMC_SAMPLES)

        return JSONResponse(
            {
                "filename": file.filename,
                "image_size": [image_height, image_width],
                "num_detections": len(detections),
                "detections": detections,
                "threat_assessment": threat_result,
                "inference": {
                    "confidence_threshold": CONF_THRESHOLD,
                    "iou_threshold": IOU_THRESHOLD,
                    "image_size": IMAGE_SIZE,
                    "tta": USE_TTA,
                },
            }
        )

    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)
