import sys
import tempfile
from pathlib import Path

import streamlit as st
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.inference import _load_pipeline, predict_quantity

MODEL_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


@st.cache_resource(show_spinner="Loading model (first time only)...")
def load_pipeline():
    pipe = _load_pipeline(str(MODEL_DIR), str(PROCESSED_DIR))
    with torch.inference_mode():
        dummy = torch.zeros(1, 3, 224, 224)
        _ = pipe["extractor"](dummy)
    return pipe


st.set_page_config(page_title="Sales Quantity Predictor", layout="centered")

st.title("Sales Quantity Predictor")
st.write("Upload a product image to estimate how many units it is likely to sell.")

load_pipeline()

uploaded = st.file_uploader(
    "Product image", type=["jpg", "jpeg", "png"],
    help="Upload a JPG, JPEG, or PNG image of the product design."
)

rate = st.number_input(
    "Expected price (Rs)", min_value=400, max_value=1700,
    value=950, step=50,
    help="The price at which you plan to sell this product. Recommended range matches the training data."
)

predict = st.button("Predict", type="primary")

if predict:
    if uploaded is None:
        st.warning("Please upload an image first.")
    else:
        suffix = Path(uploaded.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = Path(tmp.name)

        with st.spinner("Analysing image..."):
            result = predict_quantity(
                image_path=tmp_path,
                rate=float(rate),
                model_dir=MODEL_DIR,
                processed_dir=PROCESSED_DIR,
            )

        tmp_path.unlink(missing_ok=True)

        img_col, res_col = st.columns([1, 1])
        with img_col:
            st.image(uploaded, use_column_width=True)
        with res_col:
            st.metric("Predicted quantity", f"{result.point_estimate} units")
            st.write(f"Expected range: **{result.lower} – {result.upper} units**")
            st.write(f"Demand tier: **{result.tier}**")
            top_prob = result.tier_probabilities[result.tier]
            st.caption(f"Tier confidence: {int(top_prob * 100)}% "
                        f"(Low {int(result.tier_probabilities['Low'] * 100)}%, "
                        f"Medium {int(result.tier_probabilities['Medium'] * 100)}%, "
                        f"High {int(result.tier_probabilities['High'] * 100)}%)")
