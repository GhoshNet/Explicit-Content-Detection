# ContentGuard — Explicit Content Detection

An AI-powered explicit content detection system using a custom-trained **YOLOv9** model. The project ships with two interfaces: a full-stack **React + FastAPI** web application and a standalone **Streamlit** app.

**Streamlit App:** https://explicit-content-detection.streamlit.app/

---

## Features

- **Drag-and-Drop Upload** — supports JPG, PNG, and WEBP (up to 10 MB)
- **YOLOv9 Object Detection** — fine-tuned on a labeled explicit content dataset (9,273 training images)
- **Annotated Output** — bounding boxes and class labels overlaid on the uploaded image
- **Confidence Scoring** — per-detection confidence percentages with color-coded indicators
- **Safe/Unsafe Classification** — clear verdict with an alert banner
- **Two Interfaces** — modern React SPA or standalone Streamlit app

---

## Project Structure

```
Explicit-Content-Detection/
├── backend/
│   ├── main.py              # FastAPI server — POST /api/detect, GET /health
│   └── requirements.txt     # Backend Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Root component — state management & layout
│   │   ├── main.jsx         # React entry point
│   │   ├── index.css        # Global styles
│   │   └── components/
│   │       ├── Header.jsx   # Navigation bar with status badge
│   │       ├── DropZone.jsx # File upload with drag-and-drop
│   │       └── Results.jsx  # Detection results with confidence bars
│   └── package.json
├── Model/
│   └── best.pt              # Fine-tuned YOLOv9 weights
├── streamlit_app.py         # Standalone Streamlit interface
├── train.py                 # YOLOv9 training script
├── data_abs.yaml            # Dataset config (absolute paths)
├── yolo11m.pt               # Base pretrained weights
├── start.sh                 # Dev launcher — starts backend + frontend
└── requirements.txt         # Streamlit app dependencies
```

---

## Quick Start

### Option 1 — React + FastAPI (Full Stack)

**Prerequisites:** Python 3.9+, Node.js 18+

```bash
# 1. Clone the repo
git clone <repo-url>
cd Explicit-Content-Detection

# 2. Start both servers with a single script
chmod +x start.sh
./start.sh
```

Or start them manually:

```bash
# Backend (FastAPI on port 8000)
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Frontend (React + Vite on port 5173)
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

---

### Option 2 — Streamlit (Standalone)

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open the local URL printed in the terminal (default: `http://localhost:8501`).

---

## API Reference

### `POST /api/detect`

Upload an image for analysis.

**Request:** `multipart/form-data` with a `file` field (JPG / PNG / WEBP).

**Response:**
```json
{
  "is_safe": false,
  "alert_message": "⚠️ Explicit content detected: Breast, Genitalia",
  "annotated_image": "<base64-encoded PNG>",
  "detections": [
    {
      "class_name": "Breast",
      "confidence": 0.87,
      "bbox": [120, 45, 310, 290]
    }
  ]
}
```

### `GET /health`

Returns server status and model availability.

---

## Detection Classes

The model detects five classes of explicit content:

| # | Class | Description |
|---|-------|-------------|
| 0 | Breast | Exposed breast |
| 1 | Combination | Multiple explicit elements together |
| 2 | Genitalia | Explicit genitalia |
| 3 | Mouth | Explicit mouth activity |
| 4 | Penis | Explicit male anatomy |

---

## Model Training

The model is a custom-trained **YOLOv9** network, fine-tuned on the [BE Explicit v1 dataset](https://universe.roboflow.com/tanmay-pwudb/be-explicit/dataset/1) from Roboflow.

**Dataset splits:**

| Split | Images |
|-------|--------|
| Train | 9,273  |
| Val   | 866    |
| Test  | 443    |

**Training configuration:**

| Parameter | Value |
|-----------|-------|
| Base model | YOLOv9 |
| Epochs | 1000 (early stop @ 50) |
| Image size | 640 px |
| Batch size | 16 |
| Optimizer | AdamW |
| Confidence threshold | 0.5 |

To retrain:

```bash
python train.py
```

Trained weights are automatically copied to `Model/best.pt`.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Detection model | YOLOv9 |
| Backend API | FastAPI + Uvicorn |
| Frontend | React 18 + Vite + TailwindCSS |
| Standalone UI | Streamlit |
| Image processing | OpenCV, Pillow |

---

## License

This project is for educational and research purposes. The dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Review all dependency and model licenses before deploying to production.
