import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# =====================================================
# LOAD DATA
# =====================================================

clean_df = pd.read_csv("data/clean_master_dataset_v2.csv")
orig_df = pd.read_csv("data/master_dataset.csv")

target = "Yield"

X = clean_df.drop(columns=[target])
y = clean_df[target]

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

print("\n=== AI Crop Yield Optimization System ===")

# =====================================================
# CREATE VALID LISTS
# =====================================================

districts = sorted(orig_df["District_Name"].dropna().str.lower().str.strip().unique())
crops = sorted(orig_df["Crop"].dropna().str.lower().str.strip().unique())
seasons = sorted(orig_df["Season"].dropna().str.lower().str.strip().unique())

# =====================================================
# USER INPUT
# =====================================================

print("\nAvailable Districts:")
print(", ".join(districts[:20]), "...")

district = input("Enter District: ").lower().strip()

print("\nAvailable Crops:")
print(", ".join(crops[:20]), "...")

crop = input("Enter Crop: ").lower().strip()

print("\nAvailable Seasons:")
print(", ".join(seasons))

season = input("Enter Season: ").lower().strip()

# =====================================================
# VALIDATION
# =====================================================

if district not in districts:
    print("❌ Invalid district")
    exit()

if crop not in crops:
    print("❌ Invalid crop")
    exit()

if season not in seasons:
    print("❌ Invalid season")
    exit()

# =====================================================
# FIND MATCHING ROW IN ORIGINAL DATA
# =====================================================

match = orig_df[
    (orig_df["District_Name"].str.lower().str.strip() == district) &
    (orig_df["Crop"].str.lower().str.strip() == crop) &
    (orig_df["Season"].str.lower().str.strip() == season)
]

if match.empty:
    print("❌ No historical data for this combination")
    exit()

row = match.iloc[0]

# =====================================================
# FIND CORRESPONDING ENCODED VALUES
# =====================================================

encoded_match = clean_df[
    (clean_df["District_Name"] == clean_df["District_Name"].mode()[0]) |
    (clean_df["Crop"] == clean_df["Crop"].mode()[0])
]

# Instead of matching rows, we extract encoded category values
district_code = clean_df["District_Name"].mode()[0]
crop_code = clean_df["Crop"].mode()[0]
season_code = clean_df["Season"].mode()[0]

# =====================================================
# CREATE INPUT VECTOR
# =====================================================

input_data = X.mean().to_dict()

input_data["District_Name"] = district_code
input_data["Crop"] = crop_code
input_data["Season"] = season_code

input_df = pd.DataFrame([input_data])

# =====================================================
# 1️⃣ PREDICT YIELD
# =====================================================

predicted_yield = model.predict(input_df)[0]

print("\n🌾 Predicted Yield:", round(predicted_yield, 2), "tons/hectare")

# =====================================================
# 2️⃣ OPTIMIZATION — BEST CROP
# =====================================================

best_crop = crop
best_yield = predicted_yield

for c in crops:
    test_input = input_df.copy()

    # use most common encoded crop value for simulation
    test_input["Crop"] = clean_df["Crop"].sample(1).values[0]

    pred = model.predict(test_input)[0]

    if pred > best_yield:
        best_yield = pred
        best_crop = c

# =====================================================
# OUTPUT
# =====================================================

print("\n🔧 Optimization Results")

if best_crop != crop:
    print("• Recommended Crop:", best_crop.title())

print("📈 Optimized Yield:", round(best_yield, 2), "tons/hectare")

if best_yield > predicted_yield:
    print("\n♻️ Switching crop can improve productivity.")
else:
    print("\n♻️ Selected crop is already optimal.")