"""
Crop Yield Prediction — interactive CLI script (matches ai_system.py approach)
"""

import sys
import os
import pickle

# --- locate project root -----------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    import pandas as pd  # only needed at runtime, not on import

    # =====================================================
    # LOAD SAVED MODEL PACKAGE
    # =====================================================
    model_path = os.path.join(ROOT, "models", "crop_yield_model.pkl")
    if not os.path.exists(model_path):
        sys.exit(f"[ERROR] Model file not found: {model_path}\nRun train_model.py first.")

    with open(model_path, "rb") as f:  # nosec B301
        package = pickle.load(f)

    model    = package["model"]
    encoders = package["encoders"]
    features = package["features"]

    print("\n=== AI Crop Yield Prediction System ===")

    # ------------------------------------------------------------------
    # Helper: prompt user until a valid value is entered
    # ------------------------------------------------------------------
    def pick_from(label: str, enc_key: str) -> str:
        classes = list(encoders[enc_key].classes_)
        print(f"\nAvailable {label}s:")
        print(classes[:20], ("..." if len(classes) > 20 else ""))
        while True:
            val = input(f"Enter {label}: ").strip()
            if val in classes:
                return val
            print(f"  [!] '{val}' not recognised. Please choose from the list above.")

    def pick_float(prompt: str, min_val: float = 0.0, max_val: float = 1e9) -> float:
        while True:
            try:
                v = float(input(prompt).strip())
                if min_val <= v <= max_val:
                    return v
                print(f"  [!] Value must be between {min_val} and {max_val}.")
            except ValueError:
                print("  [!] Please enter a valid number.")

    # =====================================================
    # CATEGORICAL INPUTS (validated)
    # =====================================================
    district     = pick_from("District",       "District_Name")
    crop         = pick_from("Crop",           "Crop")
    season       = pick_from("Season",         "Season")
    fertility    = pick_from("Fertility Level","Fertility_Level")
    texture      = pick_from("Soil Texture",   "Texture")
    drainage     = pick_from("Drainage",       "Drainage")
    water_hold   = pick_from("Water Holding",  "Water_Holding")
    salinity     = pick_from("Salinity",       "Salinity")
    irrigation   = pick_from("Irrigation Level","Irrigation_Level")
    climate      = pick_from("Climate Type",   "Climate_Type")

    # =====================================================
    # NUMERIC INPUTS (validated)
    # =====================================================
    rainfall    = pick_float("\nRainfall (mm): ",               0, 10000)
    temp_max    = pick_float("Max Temperature (°C): ",         -10, 60)
    temp_min    = pick_float("Min Temperature (°C): ",         -20, 50)
    area        = pick_float("Area Cultivated (hectares): ",     0, 1e6)
    net_area    = pick_float("Net Cropped Area (ha): ",          0, 1e6)
    gross_area  = pick_float("Gross Cropped Area (ha): ",        0, 1e6)
    ph          = pick_float("Soil pH: ",                        0, 14)

    # =====================================================
    # BUILD INPUT ROW — same column order used at training
    # =====================================================
    input_dict = {
        "District_Name":       encoders["District_Name"].transform([district])[0],
        "Crop":                encoders["Crop"].transform([crop])[0],
        "Season":              encoders["Season"].transform([season])[0],
        "Rainfall_mm":         rainfall,
        "Temp_Max_C":          temp_max,
        "Temp_Min_C":          temp_min,
        "Area":                area,
        "pH":                  ph,
        "Fertility_Level":     encoders["Fertility_Level"].transform([fertility])[0],
        "Texture":             encoders["Texture"].transform([texture])[0],
        "Drainage":            encoders["Drainage"].transform([drainage])[0],
        "Water_Holding":       encoders["Water_Holding"].transform([water_hold])[0],
        "Salinity":            encoders["Salinity"].transform([salinity])[0],
        "Irrigation_Level":    encoders["Irrigation_Level"].transform([irrigation])[0],
        "Net_Cropped_Area_ha": net_area,
        "Gross_Cropped_Area_ha": gross_area,
        "Climate_Type":        encoders["Climate_Type"].transform([climate])[0],
    }

    # Use pd.DataFrame with named columns (avoids sklearn feature-name warnings
    # and the silent 0.0 prediction bug that occurs with raw lists)
    df_row = pd.DataFrame([input_dict], columns=features)

    # =====================================================
    # PREDICT
    # =====================================================
    prediction = model.predict(df_row)[0]
    print(f"\n*** Predicted Yield: {round(prediction, 2)} tons/hectare ***")


if __name__ == "__main__":
    main()