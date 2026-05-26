"""
🚗 Indian Vehicle Number Plate Detector
========================================
Detects Indian number plates from videos using YOLOv8 + EasyOCR.
Run with: streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import easyocr
import re
import os
import time
import tempfile
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from PIL import Image

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PlateScan India",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────
# GLOBAL CONSTANTS
# ──────────────────────────────────────────────────────────────
OUTPUT_DIR   = Path("output")
PLATES_DIR   = Path("plates")
PLATES_FILE  = Path("plates.txt")
MODEL_DIR    = Path("yolov8_model")
FRAME_SKIP   = 3          # Process every Nth frame (1 = every frame)
CONF_THRESH  = 0.35       # YOLOv8 confidence threshold

# Indian number-plate regex
# Format: 2 letters | 1-2 digits | 1-2 letters | 4 digits
# e.g. DL01AB1234 / MH12DE1433 / UK07A2089
PLATE_REGEX = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}$")

# Create dirs on startup
for d in [OUTPUT_DIR, PLATES_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# DARK-THEME  +  CYBERPUNK CSS
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;800&display=swap');

/* ── Root palette ── */
:root {
    --bg:        #080c10;
    --surface:   #0d1117;
    --border:    #1e2d40;
    --neon:      #00e5ff;
    --neon2:     #39ff14;
    --warn:      #ff6b35;
    --text:      #c9d1d9;
    --muted:     #586069;
    --card:      #111820;
}

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Exo 2', sans-serif;
}
[data-testid="stHeader"] { background: transparent !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--neon); border-radius: 2px; }

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
    position: relative;
}
.hero::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse 60% 120% at 50% -20%,
                rgba(0,229,255,.07) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: clamp(2.2rem, 5vw, 3.6rem);
    font-weight: 800;
    letter-spacing: .06em;
    text-transform: uppercase;
    background: linear-gradient(135deg, var(--neon) 0%, #7efff5 50%, var(--neon2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-shadow: none;
    margin: 0;
}
.hero-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: .9rem;
    color: var(--muted);
    margin-top: .5rem;
    letter-spacing: .15em;
}
.badge {
    display: inline-block;
    background: rgba(0,229,255,.1);
    border: 1px solid var(--neon);
    color: var(--neon);
    font-family: 'Share Tech Mono', monospace;
    font-size: .7rem;
    padding: .15rem .6rem;
    border-radius: 2px;
    margin: .25rem .15rem;
    letter-spacing: .1em;
}

/* ── Section headers ── */
.section-hdr {
    font-family: 'Share Tech Mono', monospace;
    font-size: .75rem;
    letter-spacing: .2em;
    color: var(--neon);
    text-transform: uppercase;
    border-left: 3px solid var(--neon);
    padding-left: .6rem;
    margin: 1.5rem 0 .8rem;
}

/* ── Cards ── */
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1.1rem 1.3rem;
    margin-bottom: .8rem;
    position: relative;
    overflow: hidden;
}
.card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--neon), transparent);
    opacity: .5;
}

/* ── Plate chip ── */
.plate-chip {
    display: inline-block;
    background: rgba(57,255,20,.08);
    border: 1px solid var(--neon2);
    color: var(--neon2);
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.05rem;
    padding: .3rem .9rem;
    border-radius: 4px;
    margin: .3rem .3rem;
    letter-spacing: .12em;
    text-transform: uppercase;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    background: var(--card) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 8px !important;
    padding: 1.5rem !important;
    transition: border-color .2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--neon) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, rgba(0,229,255,.15), rgba(0,229,255,.05));
    border: 1px solid var(--neon) !important;
    color: var(--neon) !important;
    font-family: 'Share Tech Mono', monospace !important;
    letter-spacing: .1em;
    text-transform: uppercase;
    border-radius: 4px !important;
    transition: all .2s;
}
.stButton > button:hover {
    background: rgba(0,229,255,.25) !important;
    box-shadow: 0 0 18px rgba(0,229,255,.35);
}

/* ── Progress bar ── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--neon), var(--neon2)) !important;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: .8rem 1rem;
}
[data-testid="stMetricValue"] {
    color: var(--neon) !important;
    font-family: 'Share Tech Mono', monospace;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; }

/* ── Alerts / info boxes ── */
[data-testid="stAlert"] {
    background: rgba(0,229,255,.06) !important;
    border-left: 3px solid var(--neon) !important;
    color: var(--text) !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Stat bar ── */
.stat-row {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin-bottom: 1rem;
}
.stat-box {
    flex: 1;
    min-width: 120px;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: .7rem 1rem;
    text-align: center;
}
.stat-val {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.6rem;
    color: var(--neon);
}
.stat-lbl {
    font-size: .68rem;
    color: var(--muted);
    letter-spacing: .15em;
    text-transform: uppercase;
}

/* ── Log console ── */
.console {
    background: #030507;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: .8rem 1rem;
    font-family: 'Share Tech Mono', monospace;
    font-size: .78rem;
    color: #4ec9b0;
    max-height: 220px;
    overflow-y: auto;
    line-height: 1.7;
}
.log-ok   { color: var(--neon2); }
.log-warn { color: var(--warn); }
.log-info { color: var(--neon); }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# HERO HEADER
# ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <p class="hero-title">⚡ PlateScan India</p>
  <p class="hero-sub">// YOLOv8 · EasyOCR · Real-time Detection Pipeline //</p>
  <div style="margin-top:.8rem">
    <span class="badge">YOLOv8</span>
    <span class="badge">EasyOCR</span>
    <span class="badge">OpenCV</span>
    <span class="badge">Streamlit</span>
    <span class="badge">IND 🇮🇳</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_yolo_model() -> YOLO:
    """
    Load YOLOv8 model.
    Priority:
      1. Custom fine-tuned model at yolov8_model/best.pt
      2. Fall back to generic YOLOv8n (detects 'car' class,
         used only for demo — replace with a plate-trained model
         for production use)
    """
    custom = MODEL_DIR / "best.pt"
    if custom.exists():
        model = YOLO(str(custom))
        return model, "custom"
    # Fallback: pretrained COCO model (detects full vehicles,
    # not ideal for plates — swap with a plate-specific model)
    model = YOLO(r"K:\NumberPlateDetectionSystemDL\yoloov8n.pt")
    return model, "coco_fallback"


@st.cache_resource(show_spinner=False)
def load_ocr() -> easyocr.Reader:
    """Initialise EasyOCR reader (English only, GPU if available)."""
    return easyocr.Reader(["en"], gpu=False)


def clean_ocr_text(raw: str) -> str:
    """
    Post-process raw OCR output:
    - Uppercase
    - Remove spaces / special chars
    - Fix common OCR substitutions (0↔O, 1↔I, etc.)
    """
    text = raw.upper().strip()
    text = re.sub(r"[^A-Z0-9]", "", text)
    # Common misreads
    substitutions = {
        "O": "0",   # only in digit positions — done via regex below
    }
    # More nuanced: fix leading "O" where a digit is expected (pos 2-3)
    # We'll rely on regex validation to filter; light cleanup here:
    text = text.replace(" ", "")
    return text


def is_valid_plate(text: str) -> bool:
    """Return True if text matches Indian number plate format."""
    return bool(PLATE_REGEX.match(text))


def load_existing_plates() -> set:
    """Load previously saved plates from plates.txt."""
    if PLATES_FILE.exists():
        with open(PLATES_FILE, "r") as f:
            return {line.strip() for line in f if line.strip()}
    return set()


def save_plate(plate: str, plates_set: set) -> bool:
    """Append new unique plate to plates.txt. Returns True if newly added."""
    if plate not in plates_set:
        plates_set.add(plate)
        with open(PLATES_FILE, "a") as f:
            f.write(plate + "\n")
        return True
    return False


def save_crop(crop: np.ndarray, plate_text: str, frame_idx: int) -> Path:
    """Save a cropped plate image to PLATES_DIR."""
    filename = PLATES_DIR / f"{plate_text}_{frame_idx}.jpg"
    cv2.imwrite(str(filename), crop)
    return filename


def draw_box(frame: np.ndarray, x1: int, y1: int,
             x2: int, y2: int, label: str, valid: bool) -> None:
    """Draw a stylised bounding box on the frame."""
    color = (57, 255, 20) if valid else (0, 200, 255)   # BGR
    # Filled semi-transparent rect
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
    # Border
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    # Corner accents (cyberpunk style)
    size = 12
    thick = 3
    for cx, cy, dx, dy in [
        (x1, y1, 1, 1), (x2, y1, -1, 1),
        (x1, y2, 1, -1), (x2, y2, -1, -1),
    ]:
        cv2.line(frame, (cx, cy), (cx + dx * size, cy), color, thick)
        cv2.line(frame, (cx, cy), (cx, cy + dy * size), color, thick)
    # Label background
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────
# CORE PROCESSING PIPELINE
# ──────────────────────────────────────────────────────────────

def process_video(
    video_path: str,
    progress_bar,
    status_text,
    log_placeholder,
    model,
    model_type: str,
    ocr_reader: easyocr.Reader,
) -> dict:
    """
    Full pipeline:
      1. Read video frame by frame
      2. YOLOv8 detect plates (or vehicles for fallback model)
      3. Crop detections
      4. EasyOCR → clean → validate regex
      5. Annotate frames & write output video
      6. Save plates + crops
    Returns summary dict.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("❌ Could not open video file.")
        return {}

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output video writer
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"annotated_{ts}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    known_plates  = load_existing_plates()
    session_plates: dict[str, Path] = {}   # plate_text → crop path
    logs: list[str] = []
    frame_idx = 0
    detected_count = 0

    def push_log(msg: str, cls: str = "") -> None:
        tag = f'<span class="log-{cls}">' if cls else "<span>"
        logs.append(f"{tag}[{frame_idx:05d}] {msg}</span>")
        log_placeholder.markdown(
            '<div class="console">' + "<br>".join(logs[-30:]) + "</div>",
            unsafe_allow_html=True,
        )

    push_log("Pipeline started", "info")
    push_log(f"Model: {model_type} | Frames: {total} | FPS: {fps:.1f}", "info")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        progress = frame_idx / max(total, 1)
        progress_bar.progress(min(progress, 1.0))
        status_text.markdown(
            f'<p style="font-family:\'Share Tech Mono\',monospace;'
            f'color:#00e5ff;font-size:.8rem;">'
            f"Processing frame {frame_idx}/{total} "
            f"({int(progress*100)}%)</p>",
            unsafe_allow_html=True,
        )

        # ── Frame skip ──
        if frame_idx % FRAME_SKIP != 0:
            writer.write(frame)
            continue

        # ── YOLOv8 inference ──
        results = model.predict(
            frame,
            conf=CONF_THRESH,
            verbose=False,
            device="cpu",   # change to "0" if CUDA available
        )

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])

                # ── Model-specific class filtering ──
                if model_type == "coco_fallback":
                    # COCO class 2 = car, 3 = motorcycle, 5 = bus, 7 = truck
                    if cls_id not in {2, 3, 5, 7}:
                        continue
                # For custom plate model: all detections are plates

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                # Guard bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width - 1, x2), min(height - 1, y2)
                if (x2 - x1) < 20 or (y2 - y1) < 8:
                    continue

                crop = frame[y1:y2, x1:x2]

                # ── OCR ──
                try:
                    ocr_results = ocr_reader.readtext(crop, detail=1)
                except Exception as e:
                    push_log(f"OCR error: {e}", "warn")
                    continue

                best_text = ""
                best_conf = 0.0
                for (_, text, ocr_conf) in ocr_results:
                    cleaned = clean_ocr_text(text)
                    if ocr_conf > best_conf and len(cleaned) >= 6:
                        best_conf  = ocr_conf
                        best_text  = cleaned

                valid   = is_valid_plate(best_text)
                label   = best_text if best_text else f"PLATE {conf:.0%}"

                # ── Annotate frame ──
                draw_box(frame, x1, y1, x2, y2, label, valid)
                detected_count += 1

                if valid and best_text not in session_plates:
                    crop_path = save_crop(crop, best_text, frame_idx)
                    session_plates[best_text] = crop_path
                    is_new = save_plate(best_text, known_plates)
                    push_log(
                        f"✅ VALID — {best_text} "
                        f"({'NEW' if is_new else 'DUPLICATE'}) "
                        f"conf={conf:.2f}",
                        "ok",
                    )
                elif best_text and not valid:
                    push_log(f"⚠ INVALID — {best_text!r}", "warn")

        writer.write(frame)

    cap.release()
    writer.release()
    push_log("Pipeline complete ✓", "ok")

    return {
        "out_path":       out_path,
        "total_frames":   frame_idx,
        "detected":       detected_count,
        "plates":         session_plates,      # plate_text → crop path
        "all_saved":      load_existing_plates(),
    }


# ──────────────────────────────────────────────────────────────
# SIDEBAR — settings
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-hdr">⚙ Settings</p>', unsafe_allow_html=True)
    FRAME_SKIP  = st.slider("Frame skip (higher = faster)", 1, 10, FRAME_SKIP)
    CONF_THRESH = st.slider("Detection confidence", 0.1, 0.9, CONF_THRESH, 0.05)
    st.markdown("---")
    st.markdown('<p class="section-hdr">📁 Saved Plates</p>', unsafe_allow_html=True)
    existing = load_existing_plates()
    if existing:
        for p in sorted(existing):
            st.code(p)
    else:
        st.caption("No plates saved yet.")
    if st.button("🗑 Clear All Plates"):
        if PLATES_FILE.exists():
            PLATES_FILE.unlink()
        for f in PLATES_DIR.iterdir():
            f.unlink()
        st.rerun()

# ──────────────────────────────────────────────────────────────
# MAIN LAYOUT
# ──────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1.1, 1], gap="large")

with col_left:
    # ── Upload ──
    st.markdown('<p class="section-hdr">📹 Upload Video</p>', unsafe_allow_html=True)
    st.markdown(
        '<div class="card">'
        "<p style='color:#586069;font-size:.8rem;margin:0 0 .5rem;'>"
        "Supported: MP4, AVI, MOV, MKV · Max: 500 MB"
        "</p></div>",
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Drop your video here",
        type=["mp4", "avi", "mov", "mkv"],
        label_visibility="collapsed",
    )

    if uploaded:
        st.video(uploaded)

with col_right:
    st.markdown('<p class="section-hdr">ℹ System Info</p>', unsafe_allow_html=True)
    st.markdown(
        '<div class="card">'
        '<p style="font-family:\'Share Tech Mono\',monospace;font-size:.8rem;'
        'color:#c9d1d9;line-height:2;">'
        "🔍 Model: YOLOv8<br>"
        "🔡 OCR: EasyOCR (EN)<br>"
        "🇮🇳 Regex: <code>^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}$</code><br>"
        "💾 Output: ./output/ &amp; ./plates/<br>"
        "📄 Log: plates.txt"
        "</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown('<p class="section-hdr">🔌 Load Model</p>', unsafe_allow_html=True)
    with st.spinner("Loading YOLOv8 + EasyOCR…"):
        try:
            yolo_model, model_type = load_yolo_model()
            ocr = load_ocr()
            model_ok = True
        except Exception as e:
            st.error(f"Model load failed: {e}")
            model_ok = False

    if model_ok:
        if model_type == "custom":
            st.success("✅ Custom plate model loaded (`yolov8_model/best.pt`)")
        else:
            st.warning(
                "⚠ Using generic YOLOv8n (COCO). "
                "For accurate results, add a plate-specific model to "
                "`yolov8_model/best.pt`."
            )

# ── Process button ──
st.markdown("---")
run_col, _ = st.columns([1, 3])
with run_col:
    run_btn = st.button("🚀  START DETECTION", use_container_width=True)

# ──────────────────────────────────────────────────────────────
# PROCESSING
# ──────────────────────────────────────────────────────────────
if run_btn:
    if not uploaded:
        st.warning("⚠ Please upload a video first.")
    elif not model_ok:
        st.error("❌ Model not loaded. Check installation.")
    else:
        # Save upload to temp file
        suffix = Path(uploaded.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        st.markdown(
            '<p class="section-hdr">⚡ Processing Pipeline</p>',
            unsafe_allow_html=True,
        )
        prog = st.progress(0)
        status_txt = st.empty()
        log_box = st.empty()

        t0 = time.time()
        summary = process_video(
            video_path=tmp_path,
            progress_bar=prog,
            status_text=status_txt,
            log_placeholder=log_box,
            model=yolo_model,
            model_type=model_type,
            ocr_reader=ocr,
        )
        elapsed = time.time() - t0
        os.unlink(tmp_path)

        if summary:
            prog.progress(1.0)
            st.success(f"✅ Done in {elapsed:.1f}s")

            # ── Stats row ──
            st.markdown(
                '<p class="section-hdr">📊 Results</p>',
                unsafe_allow_html=True,
            )
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Frames Processed", summary["total_frames"])
            c2.metric("Detections",        summary["detected"])
            c3.metric("Valid Plates",       len(summary["plates"]))
            c4.metric("Processing Time",    f"{elapsed:.1f}s")

            # ── Detected plates ──
            if summary["plates"]:
                st.markdown(
                    '<p class="section-hdr">🔢 Detected Number Plates</p>',
                    unsafe_allow_html=True,
                )
                chips = "".join(
                    f'<span class="plate-chip">{p}</span>'
                    for p in sorted(summary["plates"].keys())
                )
                st.markdown(
                    f'<div class="card">{chips}</div>',
                    unsafe_allow_html=True,
                )

                # ── Cropped plate images ──
                st.markdown(
                    '<p class="section-hdr">🖼 Cropped Plates</p>',
                    unsafe_allow_html=True,
                )
                crop_cols = st.columns(min(len(summary["plates"]), 4))
                for idx, (plate_txt, crop_path) in enumerate(
                    sorted(summary["plates"].items())
                ):
                    with crop_cols[idx % 4]:
                        if crop_path.exists():
                            img = Image.open(crop_path)
                            st.image(
                                img,
                                caption=plate_txt,
                                use_container_width=True,
                            )
            else:
                st.info(
                    "No valid Indian plates detected. "
                    "Try a plate-specific YOLOv8 model for better results."
                )

            # ── Annotated video ──
            if summary["out_path"].exists():
                st.markdown(
                    '<p class="section-hdr">🎬 Annotated Video</p>',
                    unsafe_allow_html=True,
                )
                with open(summary["out_path"], "rb") as vf:
                    st.download_button(
                        "⬇ Download Annotated Video",
                        data=vf,
                        file_name=summary["out_path"].name,
                        mime="video/mp4",
                        use_container_width=True,
                    )
                st.video(str(summary["out_path"]))

            # ── All-time plate log ──
            if summary["all_saved"]:
                st.markdown(
                    '<p class="section-hdr">📋 All Saved Plates (plates.txt)</p>',
                    unsafe_allow_html=True,
                )
                with st.expander("View full plates.txt", expanded=False):
                    st.code("\n".join(sorted(summary["all_saved"])))
