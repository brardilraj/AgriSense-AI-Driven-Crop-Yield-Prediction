# 🌾 AgriSense — Crop Yield Prediction System

> **AI-powered agricultural intelligence platform for Tamil Nadu** — predicts crop yield, recommends optimal crops, and ranks districts by predicted productivity using machine learning.

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Key Features](#-key-features)
3. [System Architecture](#-system-architecture)
4. [Tech Stack](#-tech-stack)
5. [Project Structure](#-project-structure)
6. [Dataset](#-dataset)
7. [ML Models & Performance](#-ml-models--performance)
8. [Feature Engineering](#-feature-engineering)
9. [API Reference](#-api-reference)
10. [Dashboard](#-dashboard)
11. [Installation & Setup](#-installation--setup)
12. [Running the Application](#-running-the-application)
13. [Running CLI Scripts](#-running-cli-scripts)
14. [Data Pipeline](#-data-pipeline)
15. [Security Audit](#-security-audit)
16. [Known Limitations](#-known-limitations)
17. [Future Work](#-future-work)

---

## 🎯 Project Overview

**AgriSense** is a full-stack machine-learning system designed to assist farmers, agricultural officers, and policy makers in Tamil Nadu with data-driven crop planning. The system ingests multi-year agricultural, climate, and soil datasets to:

- **Predict** the expected yield (tons/hectare) for a user-selected district, crop, and season.
- **Recommend** a better crop if one exists, based on district-season compatible historical data.
- **Rank all districts** in Tamil Nadu by predicted productivity for a given crop, season, and soil/climate profile.

The backend is a **Flask REST API** that serves predictions in real time. The frontend is a standalone **HTML/CSS/JS dashboard** (`dashboard/index.html`) that communicates with the API and presents results through animated KPI cards, interactive charts, and a district ranking table.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🔮 **Yield Prediction** | Predicts crop yield in tons/hectare for any valid district × crop × season combination |
| 🌱 **Crop Recommendation** | Compares all viable crops for the selected district + season and recommends the highest-yielding one |
| 🏆 **Best District Finder** | Iterates all Tamil Nadu districts and ranks them by predicted yield for a given crop/soil/climate profile |
| 📊 **Interactive Dashboard** | Real-time charts, animated KPI stat cards, and a sortable district ranking table |
| 📡 **REST API** | JSON endpoints for integration with external tools or automation pipelines |
| 🔒 **Security-audited** | Bandit static analysis run; `debug=True` is environment-controlled via `FLASK_DEBUG` |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Browser                             │
│              dashboard/index.html (HTML + CSS + JS)             │
└──────────────────────────┬──────────────────────────────────────┘
                           │  HTTP (JSON)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Flask REST API  (app.py)                       │
│                                                                 │
│  GET  /api/options        → dropdown data for both forms        │
│  POST /api/predict        → yield + crop recommendation         │
│  POST /api/best_district  → ranked district list                │
│                                                                 │
│  ┌──────────────────────┐   ┌──────────────────────────────┐   │
│  │   ai_system model    │   │   best_district model        │   │
│  │  (RandomForest 100)  │   │   (RandomForest 300 + pkl)   │   │
│  │  trained from        │   │   loaded from                │   │
│  │  clean_master_v2.csv │   │   crop_yield_model.pkl       │   │
│  └──────────────────────┘   └──────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Data Layer  (data/)                           │
│                                                                 │
│  master_dataset.csv              ← raw Tamil Nadu agri data     │
│  clean_master_dataset_v2.csv     ← pre-processed, encoded       │
│  final_dataset.csv               ← merged agri + soil + climate │
│  clean_rainfall.csv / clean_yield.csv ← cleaned intermediates   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

### Backend
| Component | Technology |
|---|---|
| Web Framework | Flask 3.x |
| CORS | Flask-CORS |
| ML Core | scikit-learn (RandomForestRegressor) |
| Data Manipulation | pandas, NumPy |
| Model Serialisation | pickle |
| Linting / Security | Ruff, Bandit |

### Frontend
| Component | Technology |
|---|---|
| Structure | HTML5 |
| Styling | Vanilla CSS (Fira Sans + Fira Code, CSS custom properties) |
| Logic | Vanilla JavaScript (ES2022) |
| Charts | Chart.js |
| Icons | Inline SVG |

### ML Experimentation (scripts/)
| Component | Technology |
|---|---|
| Gradient Boosting | XGBoost |
| Deep Learning | TensorFlow / Keras + scikeras |
| Model Evaluation | scikit-learn KFold CVclalr |

### Python Version
- **Python 3.10+** (tested on 3.10 / 3.11)

---

## 📁 Project Structure

```
Crop yield prediction/
│
├── app.py                          # Flask REST API — main entry point
│
├── dashboard/
│   └── index.html                  # Single-page web dashboard (self-contained)
│
├── data/
│   ├── Tamilnadu agriculture yield data.csv   # Raw source data
│   ├── master_dataset.csv                     # Merged raw dataset
│   ├── clean_master_dataset_v2.csv            # Pre-processed, LabelEncoded
│   ├── final_dataset.csv                      # Full merged feature dataset
│   ├── clean_yield.csv                        # Cleaned yield records
│   ├── clean_rainfall.csv                     # Cleaned rainfall records
│   ├── rainfall_data.csv                      # Raw rainfall data
│   ├── soil_dataset.csv                       # Soil profile data
│   └── agri_dataset.csv                       # Intermediate agri merge
│
├── models/
│   └── crop_yield_model.pkl        # Serialised RandomForest (300 trees) + LabelEncoders
│
├── scripts/
│   ├── train_model.py              # Train & save crop_yield_model.pkl
│   ├── ai_system.py                # CLI: yield prediction + crop optimisation
│   ├── predict.py                  # CLI: interactive yield predictor (uses pkl)
│   ├── recommend_crop.py           # CLI: recommend best crop for given conditions
│   ├── best_district.py            # CLI: find best district for a crop
│   ├── model_comparison.py         # 5-Fold CV comparison: LR / RF / XGB / DNN
│   ├── model_metrics.py            # Detailed cross-val metrics (R², MAE, RMSE)
│   ├── baseline_model.py           # Linear Regression baseline
│   ├── xgboost_model.py            # Standalone XGBoost training
│   ├── dnn_model.py                # Standalone Keras DNN training
│   ├── cross_validation.py         # Cross-validation utility
│   ├── define_features.py          # Feature list definition
│   ├── preprocessing.py            # General preprocessing helpers
│   ├── build_master_dataset.py     # Assemble master dataset from sources
│   ├── clean_master_dataset.py     # Clean master dataset
│   ├── merge_final.py              # Merge final feature dataset
│   ├── data_merge.py               # Data merge utility
│   ├── rainfall_cleaning.py        # Clean rainfall CSV
│   ├── yield_cleaning.py           # Clean yield CSV
│   └── train_test_split.py         # Train/test split utility
│
├── bandit_report.json              # Bandit static security analysis report
├── .ruff_cache/                    # Ruff linter cache
└── venv/                           # Python virtual environment (not committed)
```

---

## 📊 Dataset

### Source
Tamil Nadu district-level agricultural data spanning multiple crop years, merged with:
- **Yield data** — crop-wise, district-wise, season-wise production in tons and area in hectares
- **Rainfall data** — monthly and annual rainfall in mm per district
- **Soil data** — fertility level, texture, drainage, water holding capacity, salinity, pH
- **Climate data** — climate type, min/max temperature

### Key Statistics
| Attribute | Value |
|---|---|
| Geography | Tamil Nadu (all districts) |
| Granularity | District × Crop × Season |
| Seasons | Kharif, Rabi, Whole Year, Summer, Winter, Autumn |
| Raw dataset size | ~9.3 MB (`master_dataset.csv`) |
| Cleaned dataset size | ~950 KB (`clean_master_dataset_v2.csv`) |
| Full merged dataset | ~3.7 MB (`final_dataset.csv`) |

### Data Pipeline Summary

```
Raw CSVs  →  rainfall_cleaning.py  →  clean_rainfall.csv
          →  yield_cleaning.py      →  clean_yield.csv
          →  build_master_dataset.py →  master_dataset.csv
          →  clean_master_dataset.py →  clean_master_dataset_v2.csv  (LabelEncoded)
          →  merge_final.py         →  final_dataset.csv
```

---

## 🤖 ML Models & Performance

### Production Model (`crop_yield_model.pkl`)

| Attribute | Value |
|---|---|
| Algorithm | Random Forest Regressor |
| Trees | 300 estimators |
| Random State | 42 |
| Train/Test Split | 80% / 20% |
| Encoding | scikit-learn `LabelEncoder` per categorical column |

### Model Comparison (5-Fold Cross-Validation)

The `scripts/model_comparison.py` and `scripts/model_metrics.py` scripts benchmark four algorithms:

| Model | R² Score | Notes |
|---|---|---|
| **Random Forest** | ✅ Best | 300 trees, selected as production model |
| XGBoost | Strong | 350 estimators, lr=0.05 |
| Deep Neural Network | Moderate | 3-layer Keras (128→64→32), 50 epochs |
| Linear Regression | Baseline | Used as lower-bound reference |

> Run `python scripts/model_comparison.py` from the project root to reproduce the comparison with your dataset.

### API Model (app.py — ai_system)

The Flask API also trains an **in-memory RandomForest (100 trees)** on `clean_master_dataset_v2.csv` at startup for the `/api/predict` endpoint. This is intentional — it allows the application to run without requiring a separately saved `.pkl` file for the prediction route while keeping full transparency of the encoding logic.

**Crop Optimisation Logic (Production Fix)**

The naive optimisation approach (used in `ai_system.py`) swaps only the Crop code on a global mean vector, which causes Sugarcane to always "win" because its training distribution is biased toward high-input rows. The **fixed approach in `app.py`** resolves this by:

1. Restricting candidate crops to those historically cultivated in the selected district + season.
2. Computing each candidate crop's prediction vector from the **mean of its own district × season × crop rows** — ensuring apples-to-apples comparisons.

---

## ⚙️ Feature Engineering

### Input Features (17 total)

| Feature | Type | Description |
|---|---|---|
| `District_Name` | Categorical (encoded) | Tamil Nadu district |
| `Crop` | Categorical (encoded) | Crop variety |
| `Season` | Categorical (encoded) | Growing season |
| `Rainfall_mm` | Numeric | Annual rainfall in millimetres |
| `Temp_Max_C` | Numeric | Maximum temperature (°C) |
| `Temp_Min_C` | Numeric | Minimum temperature (°C) |
| `Area` | Numeric | Cultivated area (hectares) |
| `pH` | Numeric | Soil pH (0–14 scale) |
| `Fertility_Level` | Categorical (encoded) | Soil fertility classification |
| `Texture` | Categorical (encoded) | Soil texture type |
| `Drainage` | Categorical (encoded) | Soil drainage class |
| `Water_Holding` | Categorical (encoded) | Soil water retention capacity |
| `Salinity` | Categorical (encoded) | Soil salinity class |
| `Irrigation_Level` | Categorical (encoded) | Irrigation infrastructure level |
| `Net_Cropped_Area_ha` | Numeric | Net sown area (ha) |
| `Gross_Cropped_Area_ha` | Numeric | Gross cropped area (ha) |
| `Climate_Type` | Categorical (encoded) | Broad climate classification |

### Target Variable

| Variable | Unit |
|---|---|
| `Yield` | tons / hectare |

---

## 📡 API Reference

The Flask server runs on `http://127.0.0.1:5000` by default.

---

### `GET /api/options`

Returns valid dropdown values for both prediction forms.

**Response (200)**
```json
{
  "districts_ai":  ["ariyalur", "chennai", "..."],
  "crops_ai":      ["bajra", "banana", "..."],
  "seasons_ai":    ["kharif", "rabi", "..."],

  "districts_bd":  ["Ariyalur", "Chennai", "..."],
  "crops_bd":      ["Bajra", "Banana", "..."],
  "seasons_bd":    ["Kharif", "Rabi", "..."],
  "fertility":     ["High", "Low", "Medium"],
  "texture":       ["Clay", "Loam", "Sandy", "..."],
  "drainage":      ["Good", "Moderate", "Poor"],
  "water_holding": ["High", "Low", "Medium"],
  "salinity":      ["High", "Low", "Medium", "None"],
  "irrigation":    ["High", "Low", "Medium"],
  "climate":       ["Arid", "Humid", "Semi-Arid", "..."]
}
```

---

### `POST /api/predict`

Predicts yield for a district × crop × season combination and recommends an optimal crop.

**Request Body**
```json
{
  "District_Name": "chennai",
  "Crop":          "rice",
  "Season":        "kharif"
}
```

**Response (200)**
```json
{
  "predicted_yield":    2.54,
  "recommended_crop":  "Sugarcane",
  "optimized_yield":   3.87,
  "is_already_optimal": false,
  "crops_compared":    12
}
```

**Error (400)**
```json
{ "error": "Invalid district: 'xyz'" }
```

---

### `POST /api/best_district`

Ranks all Tamil Nadu districts by predicted yield for the given conditions. Returns top 5 and the full ranked list.

**Request Body**
```json
{
  "Crop":                  "Rice",
  "Season":                "Kharif",
  "Rainfall_mm":           800,
  "Temp_Max_C":            38,
  "Temp_Min_C":            22,
  "Area":                  5000,
  "pH":                    6.5,
  "Fertility_Level":       "High",
  "Texture":               "Clay",
  "Drainage":              "Good",
  "Water_Holding":         "High",
  "Salinity":              "None",
  "Irrigation_Level":      "High",
  "Net_Cropped_Area_ha":   4500,
  "Gross_Cropped_Area_ha": 4800,
  "Climate_Type":          "Humid"
}
```

> **Note:** Any omitted numeric field defaults to the column mean from `clean_master_dataset_v2.csv`. Any omitted categorical field defaults to code `0` (first class alphabetically).

**Response (200)**
```json
{
  "best_district": "Thanjavur",
  "best_yield":    4.92,
  "top_districts": [
    { "district": "Thanjavur",    "predicted_yield": 4.92 },
    { "district": "Tiruvarur",    "predicted_yield": 4.71 },
    { "district": "Nagapattinam", "predicted_yield": 4.65 },
    { "district": "Cuddalore",    "predicted_yield": 4.43 },
    { "district": "Villupuram",   "predicted_yield": 4.31 }
  ],
  "all_districts": [ "...38 entries..." ]
}
```

---

## 🖥️ Dashboard

The dashboard (`dashboard/index.html`) is a single self-contained HTML file that is served directly by Flask.

### Panels

| Panel | Description |
|---|---|
| **Predict Yield** | Dropdowns for district, crop, season → shows predicted yield, recommended crop, optimised yield, number of crops compared |
| **Find Best District** | Full form with all 17 features → shows top-5 district ranking with yield bars |
| **District Ranking Table** | Animated table populated after a Best District search |
| **KPI Stat Cards** | Animate in on page load showing headline dataset metrics |

### Typography & Design
- **Fira Sans** — body and UI text
- **Fira Code** — monospaced data/metric values
- Dark-mode first design with CSS custom properties
- SVG icons throughout (no emoji in production UI)

---

## 🚀 Installation & Setup

### Prerequisites

- Python **3.10** or higher
- `pip` (or `pip3`)
- Git (optional)

### 1. Clone / Download the Project

```bash
git clone <repository-url>
# or extract the ZIP archive
cd "Crop yield prediction"
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

**Windows (PowerShell)**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt)**
```cmd
venv\Scripts\activate.bat
```

**macOS / Linux**
```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install flask flask-cors scikit-learn pandas numpy xgboost tensorflow scikeras
```

Or if a `requirements.txt` is present:

```bash
pip install -r requirements.txt
```

> **Note:** TensorFlow and scikeras are only needed for `model_comparison.py` and `model_metrics.py`. Core application (Flask API + dashboard) runs with `flask flask-cors scikit-learn pandas numpy` only.

### 5. Ensure Required Data Files Exist

The following files must be present before running `app.py`:

```
data/clean_master_dataset_v2.csv   ← used by /api/predict
data/master_dataset.csv            ← used to build dropdown lists
models/crop_yield_model.pkl        ← used by /api/best_district
```

If `crop_yield_model.pkl` is missing, train it first:

```bash
python scripts/train_model.py
```

---

## ▶️ Running the Application

### Start the Flask Server

```bash
python app.py
```

Expected startup output:
```
[AgriSense] Loading datasets…
[AgriSense] Training ai_system model (RandomForest 100 trees)…
[AgriSense] ai_system model ready.
[AgriSense] ai_system lists: 38 districts, 87 crops, 6 seasons.
[AgriSense] best_district model ready. Features: 17
[AgriSense] Dashboard API -> http://127.0.0.1:5000  (debug=True)
```

### Open the Dashboard

Navigate to **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `FLASK_DEBUG` | `1` | Set to `0` to disable debug mode in production |

```bash
# Disable debug mode for production
set FLASK_DEBUG=0   # Windows
export FLASK_DEBUG=0  # Linux / macOS
python app.py
```

---

## 🖥️ Running CLI Scripts

All scripts must be run from the **project root** directory (where `app.py` lives).

### Train the Production Model

```bash
python scripts/train_model.py
```
Trains a `RandomForestRegressor(n_estimators=300)` on `data/master_dataset.csv` and saves the model package (model + encoders + feature list) to `models/crop_yield_model.pkl`.

---

### Interactive Yield Prediction (CLI)

```bash
python scripts/predict.py
```
Prompts for all 17 features interactively and prints the predicted yield.

---

### AI Crop Optimisation System (CLI)

```bash
python scripts/ai_system.py
```
Prompts for district, crop, and season; predicts yield; and suggests the best alternative crop.

---

### Crop Recommendation (CLI)

```bash
python scripts/recommend_crop.py
```
Prompts for district, season, and soil/climate conditions (no crop selection required). Ranks all crops by predicted yield and prints the top 3.

---

### Best District Finder (CLI)

```bash
python scripts/best_district.py
```
Prompts for crop, season, and full soil/climate profile. Iterates all districts and prints the top 5.

---

### Model Comparison (5-Fold Cross-Validation)

```bash
python scripts/model_comparison.py
```
Benchmarks Linear Regression, Random Forest, XGBoost, and a Keras DNN using 5-fold CV. Prints fold-by-fold R² scores and the final ranking.

> ⚠️ Requires TensorFlow: `pip install tensorflow scikeras`

---

### Detailed Model Metrics

```bash
python scripts/model_metrics.py
```
Same as `model_comparison.py` but also reports MAE and RMSE per model.

---

## 🔄 Data Pipeline

To rebuild all processed datasets from scratch, run the following scripts **in order** from the project root:

```bash
# Step 1 — Clean raw rainfall and yield data
python scripts/rainfall_cleaning.py
python scripts/yield_cleaning.py

# Step 2 — Merge into master dataset
python scripts/build_master_dataset.py

# Step 3 — Clean and encode master dataset
python scripts/clean_master_dataset.py

# Step 4 — Merge final feature dataset
python scripts/merge_final.py

# Step 5 — Train production model
python scripts/train_model.py
```

---

## 🔒 Security Audit

A Bandit static analysis (`bandit_report.json`) was run on the production code. Summary of findings:

| ID | File | Severity | Confidence | Finding | Status |
|---|---|---|---|---|---|
| B301 | `app.py` | MEDIUM | HIGH | `pickle.load()` on trusted local model file | ✅ Accepted — model file is local, not user-supplied |
| B201 | `app.py` | HIGH | MEDIUM | `app.run(debug=True)` | ✅ Mitigated — debug mode is controlled via `FLASK_DEBUG` env var |
| B301 | `scripts/predict.py` | MEDIUM | HIGH | `pickle.load()` on local model file | ✅ Accepted — same rationale |

**Recommendations for Production Deployment:**
- Set `FLASK_DEBUG=0` in any non-development environment.
- Run behind a WSGI server (Gunicorn / uWSGI) with a reverse proxy (Nginx).
- Validate and sanitise all user inputs at the API boundary (currently done via dropdown-constrained lists).
- Consider replacing pickle with `joblib` or `ONNX` for safer model serialisation if the model source is ever untrusted.

---

## ⚠️ Known Limitations

| Limitation | Description |
|---|---|
| **Tamil Nadu only** | The dataset covers only Tamil Nadu districts. Predictions for other states are not supported. |
| **Static soil data** | Soil profiles in the dataset are aggregate estimates — per-field soil measurements are not used. |
| **Startup time** | The API trains an in-memory RandomForest (100 trees) on startup (~10–30 seconds depending on hardware). |
| **No authentication** | The API has no authentication layer — suitable for internal/research use only. |
| **Pickle serialisation** | The model `.pkl` file is trusted as local; do not expose the model loading endpoint to untrusted inputs. |
| **Model drift** | The model is not retrained automatically. Re-run `train_model.py` when new agricultural data becomes available. |

---

## 🔭 Future Work

- [ ] Add support for more Indian states by integrating national agri datasets (APEDA, Agmarknet)
- [ ] Implement automated model retraining pipeline with MLflow experiment tracking
- [ ] Add REST API authentication (JWT / API key)
- [ ] Containerise with Docker for reproducible deployment
- [ ] Replace pickle with ONNX or joblib for safer model serialisation
- [ ] Add time-series yield forecasting using LSTM or Prophet
- [ ] Integrate satellite NDVI data for real-time crop health monitoring
- [ ] Build a mobile-responsive Progressive Web App (PWA) version

---

## 📄 License

This project is developed for academic and research purposes. All agricultural data is sourced from publicly available Tamil Nadu government agricultural records.

---

<div align="center">
  <strong>AgriSense</strong> — Intelligent Crop Planning for Tamil Nadu 🌾
</div>
