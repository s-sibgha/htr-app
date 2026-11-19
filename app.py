"""
Handwritten Text Recognition - Streamlit app
CNN trained on EMNIST Balanced (47 classes) + OpenCV contour segmentation.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # quieter TensorFlow logs (set before import)

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

# ----------------------------------------------------------------- config
MODEL_PATH = Path(__file__).resolve().parent / "emnist_cnn.keras"

# EMNIST Balanced: digits, uppercase A-Z, and 11 lowercase letters
CLASS_NAMES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt")
assert len(CLASS_NAMES) == 47

MIN_AREA = 30        # ignore contours smaller than this (noise)
PAD = 20             # padding around each character (same as the report)
MAX_SIDE = 1600      # downscale huge photos to protect the 1 GB cloud memory limit
MAX_CHARS = 200      # safety cap on number of contours processed

st.set_page_config(page_title="Handwritten Text Recognition", page_icon="✍️", layout="centered")


# ------------------------------------------------------------------ model
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    """Load the trained CNN once per server process."""
    import tensorflow as tf                       # imported lazily so the page renders fast
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH.name}")
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


# ------------------------------------------------------------ image input
def load_gray(file) -> np.ndarray:
    """Read an uploaded image as an 8-bit grayscale array (handles EXIF rotation and transparency)."""
    img = Image.open(file)
    img = ImageOps.exif_transpose(img)            # phone photos: fix orientation
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(white, rgba)  # transparent pixels -> white paper
    gray = np.array(img.convert("L"), dtype=np.uint8)

    h, w = gray.shape
    scale = MAX_SIDE / max(h, w)
    if scale < 1:
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return gray


# ------------------------------------------------------------ read_text
def read_text(gray: np.ndarray, model):
    """
    Same workflow as the report: Otsu threshold -> external contours -> sort left to right
    -> square padded canvas -> 28x28 -> CNN. Returns text plus data for display.
    """
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]  # OpenCV 3/4 safe
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > MIN_AREA]
    boxes = sorted(boxes, key=lambda b: b[0])     # left to right
    truncated = len(boxes) > MAX_CHARS
    boxes = boxes[:MAX_CHARS]

    annotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    if not boxes:
        return "", annotated, [], [], False

    chars = []
    for x, y, w, h in boxes:
        char = th[y:y + h, x:x + w]
        size = max(w, h) + PAD                    # square canvas with padding
        canvas = np.zeros((size, size), dtype=np.uint8)
        yo, xo = (size - h) // 2, (size - w) // 2
        canvas[yo:yo + h, xo:xo + w] = char
        char = cv2.resize(canvas, (28, 28), interpolation=cv2.INTER_AREA).astype("float32") / 255.0
        chars.append(char)

    batch = np.stack(chars).reshape(-1, 28, 28, 1)
    probs = np.asarray(model(batch, training=False))
    idx = probs.argmax(axis=1)
    conf = probs.max(axis=1)

    text = "".join(CLASS_NAMES[i] for i in idx)

    thickness = max(1, int(round(max(gray.shape) / 400)))
    for n, (x, y, w, h) in enumerate(boxes, start=1):
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 0), thickness)
        cv2.putText(annotated, str(n), (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4 + thickness * 0.2, (255, 0, 0), thickness)

    rows = [{"#": n, "Character": CLASS_NAMES[i], "Confidence": f"{c * 100:.1f}%"}
            for n, (i, c) in enumerate(zip(idx, conf), start=1)]
    return text, annotated, rows, chars, truncated


# --------------------------------------------------------------------- UI
st.title("✍️ Handwritten Text Recognition")
st.caption("CNN trained on EMNIST Balanced (47 classes) + OpenCV contour-based character segmentation")

with st.sidebar:
    st.header("Tips for best results")
    st.markdown(
        "- Dark ink on **plain light paper**\n"
        "- Keep characters **separate** (not touching, no cursive)\n"
        "- Photograph straight-on, even light, no shadows\n"
        "- Crop closely around the writing\n"
        "- Single line, left to right"
    )
    st.header("Limitations")
    st.markdown(
        "- No spaces between words are detected\n"
        "- Similar shapes get confused (O/0, I/1, S/5, Z/2)\n"
        "- EMNIST Balanced merges some upper/lower-case pairs "
        "(e.g. C/c, O/o, S/s), so the case may differ from what you wrote\n"
        "- English letters and digits only"
    )

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()

uploaded = st.file_uploader("Upload an image of handwriting", type=["png", "jpg", "jpeg", "bmp", "webp"])

if uploaded is None:
    st.info("Upload a PNG or JPG to begin.")
    st.stop()

try:
    gray = load_gray(uploaded)
except Exception:
    st.error("That file could not be read as an image. Please try another PNG or JPG.")
    st.stop()

with st.spinner("Reading handwriting..."):
    text, annotated, rows, chars, truncated = read_text(gray, model)

if not text:
    st.image(gray, caption="Uploaded image")
    st.warning("No characters were found. Use dark writing on a plain light background.")
    st.stop()

st.subheader("Recognised text")
st.code(text, language=None)                       # built-in copy button
st.download_button("Download as .txt", text, file_name="recognised_text.txt")

if truncated:
    st.warning(f"More than {MAX_CHARS} marks were found; only the first {MAX_CHARS} (left to right) were read.")
elif len(rows) > 60:
    st.warning("Many marks were detected. The image may contain noise, texture or shadows.")

st.image(annotated, caption="Detected characters (numbered left to right)")

with st.expander("Per-character details"):
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("28x28 inputs given to the CNN:")
    cols = st.columns(min(10, len(chars)))
    for n, c in enumerate(chars[:30]):
        cols[n % len(cols)].image(c, clamp=True, caption=rows[n]["Character"])
