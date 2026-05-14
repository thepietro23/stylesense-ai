# StyleSense AI

> See the design, sense the demand.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-EE4C2C?logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.37-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/Status-POC-blue)

**Author** — Praveen Rawal · rawalpraveen886@gmail.com
*(mobile number shared via submission email)*

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

## Evaluation Questions

### 1. How did you approach this problem?

I framed it as a supervised learning task on a small dataset (180 images, 704 transactions). The plan was:

1. **EDA** — understand sales structure, image inventory, and whether folders or filenames carried product-code information.
2. **Feature engineering** — use a pretrained ResNet50 to convert each image into a numerical embedding, then assemble a training matrix combining visual features with price.
3. **Modelling** — try Ridge, Random Forest, and XGBoost; pick the best via cross-validation and a stratified held-out test set.
4. **UI** — wrap the trained pipeline in a Streamlit app so a user can upload an image and get a prediction in under a second.

Execution was iterative. After the first round I audited the model and found two real problems: the category one-hot feature carried no signal (the model ignored it correctly, but the UI was exposing a useless control), and the model's response to price was non-monotonic in places. Both were resolved by removing the category feature, applying PCA, switching to a shallower Random Forest, and smoothing the price response at inference time.

### 2. How does your prediction system work?

```
Image (224×224 RGB)
  → ResNet50 (frozen, ImageNet pretrained)        # 2048-dim embedding
  → PCA (fit on training embeddings)              # 16 principal components
  → concat with price                             # 17 features
  → StandardScaler
  → Random Forest regressor   → predicted quantity (with ±RMSE range)
  → Random Forest classifier  → demand tier (Low / Medium / High + probabilities)
```

At inference, predictions are averaged over a small price window (±150) to dampen tree-induced non-monotonicities. The regressor uses shallow trees (`max_depth=3`, `min_samples_leaf=5`) to keep generalisation stable; the classifier uses `class_weight=balanced` because the High tier is underrepresented.

### 3. What patterns did you find in the data?

- **Highly skewed sales**: median total qty per product is 12.5, but a few products sell 100+ units. Mean is dragged up to 27.5.
- **Price–quantity correlation is weak (≈ 0.14)**. Visual features carry more signal than price alone.
- **Transactions and quantity are strongly correlated (≈ 0.85)** — products that sell often also sell in larger volumes.
- **Code prefix anomaly**: 145 of 146 product codes start with `100`; one outlier starts with `500`. The four image folders do not align with any code-prefix grouping, so the folder structure is essentially a separate categorical signal (and on inspection contributes little).
- **No filename mapping**: image filenames are timestamps with no product code embedded, so the image-to-product mapping had to be reconstructed.

### 4. Where can the system fail?

- **Out-of-distribution designs.** Images visually very different from the training set will get unreliable predictions.
- **Synthetic image–product mapping.** Targets are derived from a random stratified assignment because no real mapping is available. This caps the learnable signal.
- **Extreme prices.** Outside ₹400–₹1700, predictions are extrapolations and may behave erratically. The UI restricts input to this range.
- **Low-tier classification is weak** (F1 ≈ 0.22). The model favours Medium predictions because of class imbalance, even with re-weighting.
- **No temporal awareness.** Seasonality, festivals, and trend cycles are not modelled.
- **Tiny dataset.** 180 samples means cross-validation variance is high and a single outlier in the test split materially moves R².

### 5. If you had more time, how would you improve this system?

- **Real image–product mapping** would replace the synthetic assignment and unlock substantially more signal.
- **A larger dataset** (more designs, more sales history) would let me fine-tune ResNet50's last block rather than using frozen embeddings.
- **Temporal features**: month, festival proximity, day-of-week, and lagged demand.
- **Multi-modal inputs**: product description, fabric, color tags through a small language encoder, fused with the visual features.
- **Quantile regression** for properly calibrated uncertainty rather than a fixed ±RMSE band.
- **Active learning loop**: as real sales come in for predicted designs, retrain periodically with the feedback signal.
- **A stronger classifier on the High tier** through targeted oversampling or focal-loss boosting on the underrepresented class.

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

