import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv("data/master_dataset.csv")

# =====================================================
# SELECT BEST FEATURES
# =====================================================

features = [
    "District_Name",
    "Crop",
    "Season",
    "Rainfall_mm",
    "Temp_Max_C",
    "Temp_Min_C",
    "Area",
    "pH",
    "Fertility_Level",
    "Texture",
    "Drainage",
    "Water_Holding",
    "Salinity",
    "Irrigation_Level",
    "Net_Cropped_Area_ha",
    "Gross_Cropped_Area_ha",
    "Climate_Type"
]

target = "Yield"

df = df[features + [target]].dropna()

# =====================================================
# ENCODE CATEGORICAL VARIABLES
# =====================================================

categorical_cols = [
    "District_Name",
    "Crop",
    "Season",
    "Fertility_Level",
    "Texture",
    "Drainage",
    "Water_Holding",
    "Salinity",
    "Irrigation_Level",
    "Climate_Type"
]

encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# =====================================================
# FEATURE MATRIX
# =====================================================

X = df[features]
y = df[target]

# =====================================================
# TRAIN / TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================================
# TRAIN MODEL
# =====================================================

model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

# =====================================================
# EVALUATE
# =====================================================

y_pred = model.predict(X_test)

print("\nR² Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

rmse = mean_squared_error(y_test, y_pred) ** 0.5
print("RMSE:", rmse)

# =====================================================
# SAVE MODEL + ENCODERS
# =====================================================

model_package = {
    "model": model,
    "encoders": encoders,
    "features": features
}

with open("models/crop_yield_model.pkl", "wb") as f:
    pickle.dump(model_package, f)

print("\n✅ Research-grade model trained and saved!")