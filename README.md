# StyleSense AI

> See the design, sense the demand.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-EE4C2C?logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.37-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/Status-POC-blue)

**Author** — Praveen Rawal · rawalpraveen886@gmail.com

---

## Problem

A retailer wants to know **how many units a new product design will sell** before launching it. They only have:

- The design image
- Sales history of past products

The challenge: predict expected demand for an unseen design, using just its image.

---

## Approach

1. **Image → numbers** — pass each image through pretrained ResNet50, get a 2048-dim feature vector.
2. **Compress** — PCA reduces 2048 dims to 16 (more signal per feature given only 180 samples).
3. **Predict quantity** — Random Forest regressor maps `[16 PCA features + price]` → expected units sold.
4. **Predict tier** — Random Forest classifier outputs Low / Medium / High seller, with per-class probabilities.
5. **Smooth at inference** — predictions are averaged over a small price window to keep the response stable.

---

## Results

| Metric | Value |
|---|---|
| Regression MAE | **11.26 units** (baseline 13.78) |
| Regression R² | **0.337** |
| Classifier Accuracy | **47.2%** (random 33%) |
| Classifier F1 (macro) | **0.431** |
| Latency | ~0.5 s per image (CPU) |

---

## Setup

```bash
git clone https://github.com/thepietro23/stylesense-ai.git
cd stylesense-ai

py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1      # Windows
# source venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app/streamlit_app.py
```

Opens at http://localhost:8501. Upload an image, enter price, get prediction.

---

## Reproduce training

The 180 product images are not committed (too large). Download them and extract into `data/raw/images/{1,2,3,4}/`:

📦 https://drive.google.com/drive/folders/1PxLPhLtTpt1YR1hisvHeRZ1U-3sKV6EM

Then run notebooks in order: `01_eda.ipynb` → `02_feature_engineering.ipynb` → `03_model_training.ipynb`.

---

## Project structure

```
app/streamlit_app.py     UI
src/                     feature extractor, model, inference
notebooks/               EDA, feature engineering, training
models/                  trained artefacts (loaded by app)
data/                    sales.xlsx + processed CSVs
outputs/                 diagnostic plots
```

---

## Limitations

- Only 180 images in the training set; predictions on radically new styles are low-confidence.
- Price has weak correlation with quantity (~0.14) in the data — its influence is small and is smoothed at inference.
- Image-to-product mapping is reconstructed by stratified random assignment because filenames do not encode product codes.
