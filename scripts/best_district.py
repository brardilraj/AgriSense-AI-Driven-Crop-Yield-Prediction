import pickle

# =====================================================
# LOAD TRAINED MODEL
# =====================================================

with open("models/crop_yield_model.pkl", "rb") as f:
    package = pickle.load(f)

model = package["model"]
encoders = package["encoders"]
features = package["features"]

print("\n=== Best District for Crop System ===")

# =====================================================
# HELPER FUNCTION — SHOW FULL LIST + VALIDATE INPUT
# =====================================================

def get_valid_input(title, encoder):
    print(f"\nAvailable {title}:")
    print(", ".join(encoder.classes_))

    user_input = input(f"Enter {title}: ").strip().lower()

    for item in encoder.classes_:
        if item.lower() == user_input:
            return item

    print("❌ Invalid input. Please choose from the list.")
    exit()

# =====================================================
# USER INPUT — CROP FIXED
# =====================================================

crop = get_valid_input("Crops", encoders["Crop"])
season = get_valid_input("Seasons", encoders["Season"])
fertility = get_valid_input("Fertility Levels", encoders["Fertility_Level"])
texture = get_valid_input("Soil Texture", encoders["Texture"])
drainage = get_valid_input("Drainage Types", encoders["Drainage"])
water_holding = get_valid_input("Water Holding Levels", encoders["Water_Holding"])
salinity = get_valid_input("Salinity Levels", encoders["Salinity"])
irrigation = get_valid_input("Irrigation Levels", encoders["Irrigation_Level"])
climate = get_valid_input("Climate Types", encoders["Climate_Type"])

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
    "Crop": encoders["Crop"].transform([crop])[0],
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
# PREDICT FOR ALL DISTRICTS
# =====================================================

results = []

for district in encoders["District_Name"].classes_:

    district_code = encoders["District_Name"].transform([district])[0]

    input_dict = common_data.copy()
    input_dict["District_Name"] = district_code

    input_values = [input_dict[f] for f in features]

    predicted_yield = model.predict([input_values])[0]

    results.append((district, predicted_yield))

# =====================================================
# SORT RESULTS
# =====================================================

results.sort(key=lambda x: x[1], reverse=True)

best_district, best_yield = results[0]

# =====================================================
# OUTPUT RESULTS
# =====================================================

print("\n🏆 BEST DISTRICT FOR THIS CROP")
print("👉 Recommended District:", best_district)
print("📈 Expected Yield:", round(best_yield, 2), "tons/hectare")

print("\n🌍 TOP 5 DISTRICTS")

for i, (district, yld) in enumerate(results[:5], 1):
    print(f"{i}. {district} — {round(yld, 2)} tons/hectare")

print("\n♻️ This location maximizes productivity under given conditions.")