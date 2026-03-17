import numpy as np
from ultralytics import YOLO
import streamlit as st
from PIL import Image
from io import BytesIO

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Explicit Content Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f0f13; }
    [data-testid="stSidebar"] { background-color: #16161e; border-right: 1px solid #2a2a3a; }
    .det-card {
        background: #1e1e2e;
        border: 1px solid #2a2a3a;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.75rem;
    }
    .det-class { font-size: 1rem; font-weight: 600; color: #e2e8f0; }
    .conf-track {
        background: #2a2a3a;
        border-radius: 9999px;
        height: 8px;
        overflow: hidden;
        margin-top: 0.5rem;
    }
    .gradient-text {
        background: linear-gradient(90deg, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2rem;
        font-weight: 700;
    }
    hr { border-color: #2a2a3a; }
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Model loading ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    return YOLO("Model/best.pt")

model = None
model_error = None
try:
    model = load_model()
except Exception as exc:
    model_error = str(exc)

# ── Helper functions ───────────────────────────────────────────────────────────
CONF_THRESHOLD = 0.5

def run_detection(pil_image: Image.Image, conf: float):
    """Run YOLO inference and return (annotated PIL image, detections list)."""
    results = model.predict(pil_image, conf=conf, verbose=False)
    detections = []
    for result in results:
        for box in result.boxes:
            cls_name = result.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            bbox = [int(v) for v in box.xyxy[0]]
            detections.append({"class": cls_name, "conf": confidence, "bbox": bbox})
        # result.plot() returns BGR numpy array — convert to RGB PIL
        annotated_bgr = result.plot()
        annotated_pil = Image.fromarray(annotated_bgr[..., ::-1])
    return annotated_pil, detections

def conf_color(conf: float) -> str:
    if conf >= 0.80:
        return "#ef4444"
    if conf >= 0.60:
        return "#f97316"
    return "#eab308"

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ Explicit Content Detector")
    st.caption("AI-powered explicit content detection using YOLOv9")
    st.divider()

    st.markdown("### Settings")
    CONF_THRESHOLD = st.slider("Confidence threshold", 0.1, 0.9, CONF_THRESHOLD, 0.05,
                               help="Minimum confidence required to report a detection.")

    st.divider()
    st.markdown("### Model info")
    if model:
        st.success("Model loaded ✓")
        st.caption("Weights: `Model/best.pt`")
        st.caption("Architecture: YOLOv9")
    else:
        st.error("Model not loaded")
        if model_error:
            st.caption(f"Error: {model_error}")

    st.divider()
    st.markdown("### Detection classes")
    for cls in ["Breast", "Combination", "Genitalia", "Mouth", "Penis"]:
        st.markdown(f"- {cls}")

    st.divider()
    st.caption("For educational & research use only.")

# ── Main content ───────────────────────────────────────────────────────────────
st.markdown('<p class="gradient-text">Explicit Content Detector</p>', unsafe_allow_html=True)
st.markdown("Upload an image to scan for explicit content. Results are processed locally and never stored.")
st.divider()

if model_error or model is None:
    st.error("⚠️ Model could not be loaded. Please ensure `Model/best.pt` exists.")
    st.stop()

# ── Upload ─────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Drag and drop an image, or click to browse",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded is None:
    st.info("No image uploaded yet. Supported formats: JPG · PNG · WEBP (max 10 MB)")
    st.stop()

# ── Process ────────────────────────────────────────────────────────────────────
try:
    orig_image = Image.open(BytesIO(uploaded.getvalue())).convert("RGB")
except Exception:
    st.error("Could not decode the image. Please upload a valid JPG, PNG, or WEBP file.")
    st.stop()

with st.spinner("Analyzing image…"):
    annotated_image, detections = run_detection(orig_image, CONF_THRESHOLD)

is_safe = len(detections) == 0

# ── Verdict banner ─────────────────────────────────────────────────────────────
if is_safe:
    st.success("✅ **Safe** — No explicit content detected.")
else:
    classes_found = ", ".join(sorted({d["class"] for d in detections}))
    st.error(f"⚠️ **Explicit content detected** — {classes_found}")

st.divider()

# ── Image comparison ───────────────────────────────────────────────────────────
col_orig, col_det = st.columns(2, gap="medium")
with col_orig:
    st.markdown("#### Original Image")
    st.image(orig_image, use_container_width=True)
with col_det:
    st.markdown("#### Detection Overlay")
    st.image(annotated_image, use_container_width=True)

# ── Detection details ──────────────────────────────────────────────────────────
if detections:
    st.divider()
    st.markdown("#### Detection Details")

    m1, m2, m3 = st.columns(3)
    m1.metric("Total detections", len(detections))
    m2.metric("Unique classes", len({d["class"] for d in detections}))
    m3.metric("Highest confidence", f"{max(d['conf'] for d in detections):.0%}")

    st.markdown("")

    for det in sorted(detections, key=lambda x: x["conf"], reverse=True):
        color = conf_color(det["conf"])
        bar_width = int(det["conf"] * 100)
        x1, y1, x2, y2 = det["bbox"]
        st.markdown(f"""
        <div class="det-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span class="det-class">{det['class']}</span>
                <span style="color:{color}; font-weight:600; font-size:0.85rem;">{det['conf']:.0%}</span>
            </div>
            <div class="conf-track">
                <div style="width:{bar_width}%; height:100%; background:{color}; border-radius:9999px;"></div>
            </div>
            <div style="margin-top:0.4rem; font-size:0.78rem; color:#64748b;">
                Bounding box &nbsp;·&nbsp; x1:{x1} y1:{y1} x2:{x2} y2:{y2}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:0.8rem; color:#64748b; margin-top:0.5rem;">
        Confidence legend:
        <span style="color:#ef4444;">■</span> High (≥ 80%) &nbsp;
        <span style="color:#f97316;">■</span> Medium (60–79%) &nbsp;
        <span style="color:#eab308;">■</span> Low (&lt; 60%)
    </div>
    """, unsafe_allow_html=True)
