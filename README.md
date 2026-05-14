# StyleSense AI

**See the design, sense the demand.**

A demand prediction engine that estimates the expected sales quantity of a new product design from its image and price.

---

## Author

| | |
|---|---|
| Name | Praveen Rawal |
| Email | rawalpraveen886@gmail.com |

---

## Overview

Given a product design image and an expected unit price, StyleSense AI predicts:

1. **Quantity** — expected number of units the design will sell (regression)
2. **Range** — an uncertainty interval around the point estimate
3. **Demand tier** — `Low` / `Medium` / `High` seller classification with per-class probability

The system is trained on 180 product images and 704 historical sales transactions, using a ResNet50 visual encoder followed by PCA, a Random Forest regressor for quantity, and a Random Forest classifier (with class balancing) for the demand tier.

The full design rationale is documented in [LOGIC_AND_APPROACH.md](LOGIC_AND_APPROACH.md).

---

## Repository contents

```
.
├── README.md                       This file
├── LOGIC_AND_APPROACH.md           Detailed design and methodology
├── requirements.txt                Python dependencies
├── .gitignore
├── .streamlit/config.toml          Streamlit upload-size config
├── app/
│   └── streamlit_app.py            Web interface
├── src/
│   ├── feature_extractor.py        ResNet50 feature extractor
│   ├── model.py                    Regressor, classifier, evaluation helpers
│   └── inference.py                End-to-end prediction pipeline
├── notebooks/
│   ├── 01_eda.ipynb                Exploratory data analysis
│   ├── 02_feature_engineering.ipynb  Image embeddings + training set assembly
│   └── 03_model_training.ipynb     Model training and evaluation
├── data/
│   ├── raw/sales.xlsx              Raw sales transactions
│   └── processed/                  Cleaned CSV artefacts
├── models/                         Persisted trained artefacts
└── outputs/                        Diagnostic plots
```

---

## Setup

### Prerequisites

- Windows / macOS / Linux
- Python 3.11 (other 3.x versions may work but are not tested)
- ~2 GB free disk space (mostly for PyTorch and ResNet50 weights)

### Installation

```bash
git clone https://github.com/thepietro23/stylesense-ai.git
cd stylesense-ai

# Create virtual environment (Python 3.11)
py -3.11 -m venv venv

# Activate it
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# macOS / Linux:
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Image dataset (required for training, optional for inference)

The 180 product images are excluded from this repository to keep the clone size small. Download them from the original Google Drive link:

https://drive.google.com/drive/folders/1PxLPhLtTpt1YR1hisvHeRZ1U-3sKV6EM

Extract such that the structure becomes:

```
data/raw/images/
├── 1/    (55 images)
├── 2/    (52 images)
├── 3/    (36 images)
└── 4/    (37 images)
```

The trained model artefacts in `models/` are included in the repository, so the Streamlit application can run inference on any uploaded image without redownloading the dataset.

---

## Running the application

```bash
streamlit run app/streamlit_app.py
```

The application opens at http://localhost:8501. Upload a product image, enter an expected unit price (₹400–₹1700), and click **Predict**.

---

## Reproducing the training pipeline

If you wish to retrain from scratch (requires the image dataset extracted as above):

```bash
jupyter notebook
```

Run the notebooks in order:

1. **01_eda.ipynb** — sales data structure, distributions, image inventory.
2. **02_feature_engineering.ipynb** — ResNet50 embeddings, image–product mapping, feature matrix.
3. **03_model_training.ipynb** — train regressor and classifier, evaluate, persist artefacts.

The training notebook overwrites the contents of `models/`.

---

## Results

### Regression (quantity prediction)

| Metric | Value |
|---|---|
| Test MAE | 11.26 units |
| Test RMSE | 15.04 units |
| Test R² | 0.337 |
| CV MAE (5-fold) | 12.83 ± 2.95 |
| Mean-only baseline MAE | 13.78 |

The regressor reduces prediction error by approximately 18 percent relative to a mean-only baseline.

### Classification (demand tier)

| Metric | Value |
|---|---|
| Accuracy | 47.2 percent |
| F1 (macro) | 0.431 |
| F1 (Medium) | 0.571 |
| F1 (High) | 0.500 |
| F1 (Low) | 0.222 |

Random guessing on three balanced classes would achieve approximately 33 percent accuracy and 0.33 F1.

### Latency

End-to-end prediction averages ~0.5 seconds per image on CPU, with a one-time model load of ~3–5 seconds on application start.

---

## Five required answers

The complete responses to the five mandatory questions (approach, model logic, observed patterns, failure modes, future improvements) are in [LOGIC_AND_APPROACH.md](LOGIC_AND_APPROACH.md).

---

## Limitations

- Only 180 product images are available for training, and the image-to-product mapping is reconstructed through stratified random assignment because filenames do not encode product codes.
- The price feature has a weak correlation with quantity in the data (~0.14), so the regressor's response to price is small and can be locally non-monotonic; inference applies a smoothing window to keep the response stable.
- Predictions on designs that are visually very different from the training distribution should be treated as low-confidence.
- The system is trained on CPU with no GPU acceleration. Inference is fast enough for interactive use, but very large batches would benefit from GPU.
