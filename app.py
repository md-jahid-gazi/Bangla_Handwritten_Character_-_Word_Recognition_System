from pathlib import Path
import html
import json

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "model.keras"
LABELS_PATH = ROOT / "labels.json"
MAP_PATH = ROOT / "unicode_map.json"

st.set_page_config(page_title="Bangla Handwritten Word Recognition", page_icon="✍️", layout="centered")
st.title("✍️ Bangla Handwritten Word Recognition")
st.caption("Write a Bengali character or short word and get a Unicode prediction.")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def preprocess_character(crop):
    """Normalize a canvas crop to the same black-ink/white-background 64x64 format as training."""
    crop = ImageOps.grayscale(crop)
    arr = np.asarray(crop)
    ys, xs = np.where(arr < 245)
    if len(xs) == 0:
        return None
    crop = crop.crop((
        max(0, int(xs.min()) - 3),
        max(0, int(ys.min()) - 3),
        min(crop.width, int(xs.max()) + 4),
        min(crop.height, int(ys.max()) + 4),
    ))
    crop.thumbnail((56, 56), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (64, 64), 255)
    canvas.paste(crop, ((64 - crop.width) // 2, (64 - crop.height) // 2))
    return np.asarray(canvas, dtype=np.uint8)[None, :, :, None]


def segment_word(image):
    """Best-effort left-to-right segmentation based on vertical whitespace."""
    arr = np.asarray(ImageOps.grayscale(image))
    ink = arr < 245
    ys, xs = np.where(ink)
    if len(xs) == 0:
        return []

    roi = ink[
        max(0, ys.min() - 4):min(arr.shape[0], ys.max() + 5),
        max(0, xs.min() - 4):min(arr.shape[1], xs.max() + 5),
    ]
    active = roi.sum(axis=0) > 0

    # Close only very small gaps; broad gaps remain character boundaries.
    if len(active) > 2:
        original = active.copy()
        active[1:-1] |= (~original[1:-1] & original[:-2] & original[2:])

    spans = []
    start = None
    for x, value in enumerate(active):
        if value and start is None:
            start = x
        elif not value and start is not None:
            if x - start >= 2:
                spans.append((start, x))
            start = None
    if start is not None and len(active) - start >= 2:
        spans.append((start, len(active)))

    # Convert boolean ink mask to black ink on white background.
    return [
        Image.fromarray((~roi[:, left:right] * 255).astype(np.uint8), mode="L")
        for left, right in spans
    ]


def folder_id_for_index(labels, class_index):
    index_to_class = labels.get("index_to_class", {})
    folder_id = index_to_class.get(str(class_index))
    if folder_id is not None:
        return str(folder_id)

    class_names = labels.get("class_names", [])
    if class_index < len(class_names):
        return str(class_names[class_index])
    raise ValueError(f"Model output index {class_index} is missing from labels.json")


canvas = st_canvas(
    fill_color="rgba(255,255,255,1)",
    stroke_width=8,
    stroke_color="#111111",
    background_color="#FFFFFF",
    height=220,
    width=700,
    drawing_mode="freedraw",
    key="bangla_canvas",
)
st.caption("Write from left to right. Leave a small gap between characters when possible.")

if st.button("Predict Unicode", type="primary", use_container_width=True):
    if not MODEL_PATH.is_file():
        st.error("Model not found. Run `python train.py` first.")
    elif not LABELS_PATH.is_file() or not MAP_PATH.is_file():
        st.error("labels.json or unicode_map.json is missing.")
    elif canvas.image_data is None:
        st.warning("Please write a character or word first.")
    else:
        try:
            labels = load_json(LABELS_PATH)
            mapping_data = load_json(MAP_PATH)
            # unicode_map.json stores the actual class-to-Unicode entries
            # inside its top-level "mapping" object.
            mapping = mapping_data.get("mapping", mapping_data)
            if not isinstance(mapping, dict):
                raise ValueError(
                    "Invalid unicode_map.json format: expected a 'mapping' object."
                )
            model = load_model()

            # Canvas already has black handwriting on a white background.
            drawing = Image.fromarray(canvas.image_data.astype(np.uint8)).convert("RGB")
            segments = segment_word(drawing)
            predicted_chars = []
            confidences = []

            for segment in segments:
                batch = preprocess_character(segment)
                if batch is None:
                    continue

                probabilities = model.predict(batch, verbose=0)[0]
                class_index = int(np.argmax(probabilities))
                folder_id = folder_id_for_index(labels, class_index)
                character = mapping.get(str(folder_id), "□")
                predicted_chars.append(character if character else "□")
                confidences.append(float(probabilities[class_index]))

            if not predicted_chars:
                st.warning("No handwriting detected. Please write again.")
            else:
                predicted_word = "".join(predicted_chars)
                avg_confidence = float(np.mean(confidences))
                st.subheader("Predicted Unicode word")
                st.markdown(
                    f"""
                    <div style="padding:24px 16px;border:1px solid #bdbdbd;
                                border-radius:12px;text-align:center;background:white;">
                      <div style="font-size:44px;font-weight:700;line-height:1.6;">
                        {html.escape(predicted_word)}
                      </div>
                      <div style="font-size:14px;margin-top:8px;">
                        Estimated character confidence: {avg_confidence * 100:.2f}%
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if "□" in predicted_word:
                    st.warning("A predicted class has no Unicode mapping (□).")
        except Exception as exc:
            st.error(f"Prediction error: {exc}")

st.caption(
    "This model is trained on isolated characters. Word output is a best-effort "
    "sequence of segmented character predictions; connected letters, vowel signs "
    "and conjunct placement can affect accuracy."
)
