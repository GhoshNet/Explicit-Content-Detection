import os
import base64
import traceback
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(title="ContentGuard API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent.parent / "Model" / "best.pt"
model = None


def load_model():
    global model
    if not ULTRALYTICS_AVAILABLE:
        print("[WARN] ultralytics not installed — model will not be loaded.")
        return
    if not MODEL_PATH.exists():
        print(f"[WARN] Model file not found at {MODEL_PATH} — running without model.")
        return
    try:
        model = YOLO(str(MODEL_PATH))
        print(f"[INFO] YOLOv9 model loaded from {MODEL_PATH}")
    except Exception as exc:
        print(f"[ERROR] Failed to load model: {exc}")
        model = None


@app.on_event("startup")
async def startup_event():
    load_model()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def image_to_base64(img_bgr: np.ndarray) -> str:
    """Encode a BGR numpy image to a base64 PNG string."""
    success, buffer = cv2.imencode(".png", img_bgr)
    if not success:
        raise RuntimeError("Failed to encode image to PNG")
    return base64.b64encode(buffer).decode("utf-8")


def draw_detections(img_bgr: np.ndarray, detections: list) -> np.ndarray:
    """Draw bounding boxes and labels on a copy of the image."""
    annotated = img_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = f"{det['class_name']} {det['confidence']:.0%}"

        # Blue bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 100, 0), 2)

        # Red label text with dark background for readability
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        bg_y1 = max(y1 - th - baseline - 4, 0)
        cv2.rectangle(annotated, (x1, bg_y1), (x1 + tw + 4, y1), (30, 30, 30), -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 2, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 220),  # red text (BGR)
            2,
            cv2.LINE_AA,
        )
    return annotated


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
    }


@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    # Validate content type loosely
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    raw = await file.read()
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    # Decode image
    arr = np.frombuffer(raw, np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image. Make sure it is a valid JPEG/PNG/WEBP.")

    detections = []

    if model is None:
        # Return the original image with a warning when the model isn't available
        annotated_b64 = image_to_base64(img_bgr)
        return JSONResponse(
            content={
                "annotated_image": annotated_b64,
                "detections": [],
                "is_safe": True,
                "alert_message": "Model not loaded — unable to perform detection. Please check server logs.",
            }
        )

    try:
        results = model.predict(source=img_bgr, conf=0.5, verbose=False)
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Model inference failed: {exc}")

    # Parse results
    if results and results[0].boxes is not None:
        boxes = results[0].boxes
        names = model.names  # class index → name dict

        for box in boxes:
            cls_idx = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = (int(v) for v in xyxy)
            class_name = names.get(cls_idx, str(cls_idx))

            detections.append(
                {
                    "class_name": class_name,
                    "confidence": round(conf, 4),
                    "bbox": [x1, y1, x2, y2],
                }
            )

    annotated_img = draw_detections(img_bgr, detections)
    annotated_b64 = image_to_base64(annotated_img)

    is_safe = len(detections) == 0
    if is_safe:
        alert_message = "No explicit content detected. Image appears safe."
    else:
        classes_found = list({d["class_name"] for d in detections})
        alert_message = (
            f"Explicit content detected: {', '.join(classes_found)}. "
            f"{len(detections)} object(s) found."
        )

    return JSONResponse(
        content={
            "annotated_image": annotated_b64,
            "detections": detections,
            "is_safe": is_safe,
            "alert_message": alert_message,
        }
    )
