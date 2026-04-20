"""
Flask API for Crop Yield Prediction Dashboard

Backend logic sources:
  - /api/predict      → replicates ai_system.py exactly
  - /api/best_district → replicates best_district.py exactly  (with DataFrame fix)
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from sklearn.ensemble import RandomForestRegressor
import logging
import pickle
import os
import sys
import pandas as pd
import numpy as np

app = Flask(__name__, static_folder="dashboard", static_url_path="")
CORS(app)

BASE = os.path.dirname(os.path.abspath(__file__))
logging.basicConfig(level=logging.INFO)

# ═══════════════════════════════════════════════════════════════════
# SECTION A — ai_system.py MODEL
#   Exactly mirrors ai_system.py:
#     • trains RandomForest on clean_master_dataset_v2.csv
#     • stores X (feature matrix), X.mean() as base vector
#     • stores mode-encoded categorical codes
#     • stores lowercase district/crop/season lists from master_dataset.csv
# ═══════════════════════════════════════════════════════════════════

CLEAN_PATH  = os.path.join(BASE, "data", "clean_master_dataset_v2.csv")
MASTER_PATH = os.path.join(BASE, "data", "master_dataset.csv")

for path in (CLEAN_PATH, MASTER_PATH):
    if not os.path.exists(path):
        sys.exit(f"[ERROR] Required data file not found: {path}")

print("[AgriSense] Loading datasets…")
_clean_df = pd.read_csv(CLEAN_PATH)
_orig_df  = pd.read_csv(MASTER_PATH, low_memory=False)

AI_TARGET = "Yield"
AI_X = _clean_df.drop(columns=[AI_TARGET])
AI_y = _clean_df[AI_TARGET]

print("[AgriSense] Training ai_system model (RandomForest 100 trees)…")
ai_model = RandomForestRegressor(n_estimators=100, random_state=42)
ai_model.fit(AI_X, AI_y)
print("[AgriSense] ai_system model ready.")

# ── Base vector = feature means (ai_system.py line 99) ─────────────
AI_MEAN = AI_X.mean().to_dict()

# ── Valid lowercase lists — same sort order LabelEncoder uses ──────
# clean_master_dataset_v2.csv was produced by LabelEncoder, which assigns
# code = alphabetical rank.  So:  code(name) = sorted_list.index(name)
AI_DISTRICTS = sorted(_orig_df["District_Name"].dropna().str.lower().str.strip().unique().tolist())
AI_CROPS     = sorted(_orig_df["Crop"].dropna().str.lower().str.strip().unique().tolist())
AI_SEASONS   = sorted(_orig_df["Season"].dropna().str.lower().str.strip().unique().tolist())

# Name → integer code lookup (fixes the mode-code bug and the cyclic-index bug)
AI_DISTRICT_CODE = {name: idx for idx, name in enumerate(AI_DISTRICTS)}
AI_CROP_CODE     = {name: idx for idx, name in enumerate(AI_CROPS)}
AI_SEASON_CODE   = {name: idx for idx, name in enumerate(AI_SEASONS)}

print(f"[AgriSense] ai_system lists: {len(AI_DISTRICTS)} districts, "
      f"{len(AI_CROPS)} crops, {len(AI_SEASONS)} seasons.")

# ═══════════════════════════════════════════════════════════════════
# SECTION B — best_district.py MODEL (pickle)
#   Exactly mirrors best_district.py:
#     • loads crop_yield_model.pkl (model + encoders + features)
#     • encodes user inputs via LabelEncoders
#     • iterates all districts, predicts, sorts
# ═══════════════════════════════════════════════════════════════════

MODEL_PATH = os.path.join(BASE, "models", "crop_yield_model.pkl")
if not os.path.exists(MODEL_PATH):
    sys.exit(f"[ERROR] Model not found: {MODEL_PATH}. Run train_model.py first.")

with open(MODEL_PATH, "rb") as f:  # nosec B301
    _pkg = pickle.load(f)

bd_model    = _pkg["model"]
bd_encoders = _pkg["encoders"]
bd_features = _pkg["features"]

print(f"[AgriSense] best_district model ready. Features: {len(bd_features)}")


# ═══════════════════════════════════════════════════════════════════
# HELPER — safe prediction via pd.DataFrame (fixes best_district
#           plain-list bug while keeping identical feature values)
# ═══════════════════════════════════════════════════════════════════

def safe_predict_bd(input_dict: dict) -> float:
    """Build a DataFrame row in correct feature order and call bd_model.predict."""
    row = {f: input_dict[f] for f in bd_features}
    df_row = pd.DataFrame([row], columns=bd_features)
    return float(bd_model.predict(df_row)[0])


# ═══════════════════════════════════════════════════════════════════
# ROUTES — static
# ═══════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory("dashboard", "index.html")


# ── Options — dropdown population ─────────────────────────────────

@app.route("/api/options", methods=["GET"])
def get_options():
    """
    Returns valid dropdown choices for both prediction forms.
    Predict Yield uses ai_system lowercase lists.
    Best District uses pickle encoder classes.
    """
    return jsonify({
        # Predict Yield: lowercase, from master_dataset
        "districts_ai":  AI_DISTRICTS,
        "crops_ai":      AI_CROPS,
        "seasons_ai":    AI_SEASONS,

        # Best District: from pickle encoders
        "districts_bd":  sorted(bd_encoders["District_Name"].classes_.tolist()),
        "crops_bd":      sorted(bd_encoders["Crop"].classes_.tolist()),
        "seasons_bd":    sorted(bd_encoders["Season"].classes_.tolist()),
        "fertility":     sorted(bd_encoders["Fertility_Level"].classes_.tolist()),
        "texture":       sorted(bd_encoders["Texture"].classes_.tolist()),
        "drainage":      sorted(bd_encoders["Drainage"].classes_.tolist()),
        "water_holding": sorted(bd_encoders["Water_Holding"].classes_.tolist()),
        "salinity":      sorted(bd_encoders["Salinity"].classes_.tolist()),
        "irrigation":    sorted(bd_encoders["Irrigation_Level"].classes_.tolist()),
        "climate":       sorted(bd_encoders["Climate_Type"].classes_.tolist()),
    })


# ═══════════════════════════════════════════════════════════════════
# ROUTE 1 — PREDICT YIELD  (exact ai_system.py logic)
#
# ai_system.py steps replicated:
#   1. Validate district/crop/season (lowercase) against orig_df lists
#   2. Check matching row exists in orig_df
#   3. input_data = X.mean()  (AI_MEAN)
#   4. Substitute mode-encoded codes for district/crop/season
#   5. Predict with ai_model
#   6. Optimization: iterate crops, swap in a sample crop code, find best
#   7. Return predicted_yield + recommended_crop + optimized_yield
# ═══════════════════════════════════════════════════════════════════

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    try:
        district = data.get("District_Name", "").strip().lower()
        crop     = data.get("Crop", "").strip().lower()
        season   = data.get("Season", "").strip().lower()

        # Step 1: validate against ai_system lowercase lists
        if district not in AI_DISTRICTS:
            return jsonify({"error": f"Invalid district: '{district}'"}), 400
        if crop not in AI_CROPS:
            return jsonify({"error": f"Invalid crop: '{crop}'"}), 400
        if season not in AI_SEASONS:
            return jsonify({"error": f"Invalid season: '{season}'"}), 400

        # Step 2: check matching row exists in orig_df
        match = _orig_df[
            (_orig_df["District_Name"].str.lower().str.strip() == district) &
            (_orig_df["Crop"].str.lower().str.strip() == crop) &
            (_orig_df["Season"].str.lower().str.strip() == season)
        ]
        if match.empty:
            return jsonify({
                "error": f"No historical data found for: {district} / {crop} / {season}"
            }), 400

        # Steps 3-4: mean base vector + user's ACTUAL encoded values
        # FIX: use AI_DISTRICT_CODE[district] not mode — each user selection
        #      now gets its correct alphabetical integer code.
        input_data = dict(AI_MEAN)
        input_data["District_Name"] = AI_DISTRICT_CODE[district]
        input_data["Crop"]          = AI_CROP_CODE[crop]
        input_data["Season"]        = AI_SEASON_CODE[season]

        input_df = pd.DataFrame([input_data], columns=list(AI_X.columns))

        # Step 5: predict yield for selected crop
        predicted_yield = float(ai_model.predict(input_df)[0])

        # Step 6: Real crop optimization
        #
        # Root cause of the "always Sugarcane" bug:
        #   - Using global X.mean() as the base vector and only swapping Crop
        #     causes the model to compare incompatible conditions: Sugarcane was
        #     trained on high-input rows so it always wins on a neutral base.
        #
        # Fix in two parts:
        #   A) Only test crops that appear in orig_df for this district+season
        #      (crops actually viable here — no exotic outliers).
        #   B) For each candidate crop c, build its prediction vector from the
        #      mean of clean_df rows that match district+season+crop.
        #      This makes comparisons "crop c under typical conditions for c"
        #      rather than "crop c under average-of-all-crops conditions".

        # A) Valid crops for this district+season
        valid_crops = sorted(
            _orig_df[
                (_orig_df["District_Name"].str.lower().str.strip() == district) &
                (_orig_df["Season"].str.lower().str.strip() == season)
            ]["Crop"].dropna().str.lower().str.strip().unique().tolist()
        )

        best_crop  = crop
        best_yield = predicted_yield

        dist_code   = AI_DISTRICT_CODE[district]
        season_code = AI_SEASON_CODE[season]

        for c in valid_crops:
            if c not in AI_CROP_CODE:
                continue
            c_code = AI_CROP_CODE[c]

            # B) Mean feature vector for crop c in this district+season
            c_rows = AI_X[
                (AI_X["District_Name"] == dist_code) &
                (AI_X["Season"]        == season_code) &
                (AI_X["Crop"]          == c_code)
            ]

            if c_rows.empty:
                # Fallback: district+season mean, swap crop code only
                test_vec = dict(input_data)
                test_vec["Crop"] = c_code
            else:
                test_vec = c_rows.mean().to_dict()
                # Lock district & season so we compare the same location
                test_vec["District_Name"] = dist_code
                test_vec["Season"]        = season_code
                test_vec["Crop"]          = c_code

            test_df = pd.DataFrame([test_vec], columns=list(AI_X.columns))
            pred = float(ai_model.predict(test_df)[0])

            if pred > best_yield:
                best_yield = pred
                best_crop  = c

        is_already_optimal = (best_crop == crop)

        return jsonify({
            "predicted_yield":    round(predicted_yield, 2),
            "recommended_crop":   best_crop.title(),
            "optimized_yield":    round(best_yield, 2),
            "is_already_optimal": is_already_optimal,
            "crops_compared":     len(valid_crops),
        })

    except Exception as e:
        app.logger.exception("Predict error: %s", e)
        return jsonify({"error": str(e)}), 400


# ═══════════════════════════════════════════════════════════════════
# ROUTE 2 — BEST DISTRICT  (exact best_district.py logic)
#
# best_district.py steps replicated:
#   1. Encode crop, season, fertility, texture, drainage, water_holding,
#      salinity, irrigation, climate via bd_encoders
#   2. Take numeric inputs: rainfall, temp_max, temp_min, area, net_area,
#      gross_area, ph
#   3. For each district in bd_encoders["District_Name"].classes_:
#        encode district → build input_dict → predict via pd.DataFrame
#   4. Sort by yield descending → return top 5 + best
# ═══════════════════════════════════════════════════════════════════

@app.route("/api/best_district", methods=["POST"])
def best_district():
    data = request.get_json()
    try:
        # Helper to get valid encoded value via pickle LabelEncoder
        # FIX: fallback uses pickle encoder's own classes (not clean_df's
        #      separate integer encoding which uses different codes).
        def enc(key: str) -> int:
            val = data.get(key, "")
            if isinstance(val, str):
                val = val.strip()
            else:
                val = ""
            if val and val in bd_encoders[key].classes_:
                return int(bd_encoders[key].transform([val])[0])
            # fallback: use code 0 (first class alphabetically)
            return 0

        def num(key: str, default: float) -> float:
            val = data.get(key)
            if val is None or val == "":
                return default
            try:
                return float(val)
            except (ValueError, TypeError):
                return default

        # Numeric means as fallbacks (same columns as best_district.py)
        _num_cols = ["Rainfall_mm", "Temp_Max_C", "Temp_Min_C",
                     "Area", "pH", "Net_Cropped_Area_ha", "Gross_Cropped_Area_ha"]
        _means = {}
        for col in _num_cols:
            if col in _clean_df.columns:
                _means[col] = float(_clean_df[col].mean())
            else:
                _means[col] = 0.0

        # Build the common data dict (analogous to best_district.py lines 63-80)
        common_data = {
            "Crop":                  enc("Crop"),
            "Season":                enc("Season"),
            "Rainfall_mm":           num("Rainfall_mm",           _means["Rainfall_mm"]),
            "Temp_Max_C":            num("Temp_Max_C",            _means["Temp_Max_C"]),
            "Temp_Min_C":            num("Temp_Min_C",            _means["Temp_Min_C"]),
            "Area":                  num("Area",                  _means["Area"]),
            "pH":                    num("pH",                    _means["pH"]),
            "Fertility_Level":       enc("Fertility_Level"),
            "Texture":               enc("Texture"),
            "Drainage":              enc("Drainage"),
            "Water_Holding":         enc("Water_Holding"),
            "Salinity":              enc("Salinity"),
            "Irrigation_Level":      enc("Irrigation_Level"),
            "Net_Cropped_Area_ha":   num("Net_Cropped_Area_ha",  _means["Net_Cropped_Area_ha"]),
            "Gross_Cropped_Area_ha": num("Gross_Cropped_Area_ha",_means["Gross_Cropped_Area_ha"]),
            "Climate_Type":          enc("Climate_Type"),
        }

        # Iterate all districts (best_district.py lines 86-99)
        results = []
        for district in bd_encoders["District_Name"].classes_:
            district_code = int(bd_encoders["District_Name"].transform([district])[0])
            input_dict = dict(common_data)
            input_dict["District_Name"] = district_code
            y_pred = safe_predict_bd(input_dict)
            results.append({"district": district, "predicted_yield": round(y_pred, 2)})

        results.sort(key=lambda x: x["predicted_yield"], reverse=True)

        return jsonify({
            "best_district": results[0]["district"],
            "best_yield":    results[0]["predicted_yield"],
            "top_districts": results[:5],
            "all_districts": results,
        })

    except Exception as e:
        app.logger.exception("Best-district error: %s", e)
        return jsonify({"error": str(e)}), 400


# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "1") == "1"
    print(f"[AgriSense] Dashboard API -> http://127.0.0.1:5000  (debug={debug_mode})")
    app.run(debug=debug_mode, port=5000)
