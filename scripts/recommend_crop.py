import pickle
import pandas as pd

# =====================================================
# LOAD TRAINED MODEL
# =====================================================

with open("models/crop_yield_model.pkl", "rb") as f:
    package = pickle.load(f)

model = package["model"]
encoders = package["encoders"]
features = package["features"]

print("\n=== AI Crop Recommendation System ===")

# =====================================================
# HELPER FUNCTION (case-insensitive input)
# =====================================================

def get_valid_input(prompt, encoder):
    print(list(encoder.classes_)[:20], "...")
    user_input = input(prompt).strip().lower()

    for item in encoder.classes_:
        if item.lower() == user_input:
            return item

    print("❌ Invalid input")
    exit()

# =====================================================
# USER INPUT (NO CROP ASKED)
# =====================================================

district = get_valid_input("\nEnter District: ", encoders["District_Name"])
season = get_valid_input("\nEnter Season: ", encoders["Season"])
fertility = get_valid_input("\nEnter Fertility Level: ", encoders["Fertility_Level"])
texture = get_valid_input("\nEnter Soil Texture: ", encoders["Texture"])
drainage = get_valid_input("\nEnter Drainage: ", encoders["Drainage"])
water_holding = get_valid_input("\nEnter Water Holding: ", encoders["Water_Holding"])
salinity = get_valid_input("\nEnter Salinity: ", encoders["Salinity"])
irrigation = get_valid_input("\nEnter Irrigation Level: ", encoders["Irrigation_Level"])
climate = get_valid_input("\nEnter Climate Type: ", encoders["Climate_Type"])

# =====================================================
# NUMERIC INPUTS
# =====================================================

rainfall = float(input("\nRainfall (mm): "))
temp_max = float(input("Max Temperature (°C): "))
temp_min = float(input("Min Temperature (°C): "))
area = float(input("Area Cultivated (hectares): "))
net_area = float(input("Net Cropped Area (ha): "))
gross_area = float(input("Gross Cropped Area (ha): "))
ph = float(input("Soil pH (4.5–8.5 typical): "))

# =====================================================
# ENCODE COMMON FEATURES
# =====================================================

common_data = {
    "District_Name": encoders["District_Name"].transform([district])[0],
    "Season": encoders["Season"].transform([season])[0],
    "Rainfall_mm": rainfall,
    "Temp_Max_C": temp_max,
    "Temp_Min_C": temp_min,
    "Area": area,
    "pH": ph,
    "Fertility_Level": encoders["Fertility_Level"].transform([fertility])[0],
    "Texture": encoders["Texture"].transform([texture])[0],
    "Drainage": encoders["Drainage"].transform([drainage])[0],
    "Water_Holding": encoders["Water_Holding"].transform([water_holding])[0],
    "Salinity": encoders["Salinity"].transform([salinity])[0],
    "Irrigation_Level": encoders["Irrigation_Level"].transform([irrigation])[0],
    "Net_Cropped_Area_ha": net_area,
    "Gross_Cropped_Area_ha": gross_area,
    "Climate_Type": encoders["Climate_Type"].transform([climate])[0],
}

# =====================================================
# PREDICT FOR ALL CROPS
# =====================================================

results = []

for crop in encoders["Crop"].classes_:

    crop_code = encoders["Crop"].transform([crop])[0]

    input_dict = common_data.copy()
    input_dict["Crop"] = crop_code

    input_values = [input_dict[f] for f in features]

    predicted_yield = model.predict([input_values])[0]

    results.append((crop, predicted_yield))

# =====================================================
# SORT BY YIELD
# =====================================================

results.sort(key=lambda x: x[1], reverse=True)

best_crop, best_yield = results[0]

# =====================================================
# OUTPUT RESULTS
# =====================================================

print("\n🌾 BEST CROP RECOMMENDATION")
print("👉 Recommended Crop:", best_crop)
print("📈 Expected Yield:", round(best_yield, 2), "tons/hectare")

print("\n🏆 TOP 3 CROPS")

for i, (crop, yld) in enumerate(results[:3], 1):
    print(f"{i}. {crop} — {round(yld, 2)} tons/hectare")

print("\n♻️ This recommendation maximizes productivity under given conditions.")