import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("data/master_dataset.csv")

print("Original shape:", df.shape)

# ---------------------------------------------------
# 1. Fix duplicate soil columns
# ---------------------------------------------------

if "Soil_Type_y" in df.columns:
    df["Soil_Type"] = df["Soil_Type_y"]
elif "Soil_Type_x" in df.columns:
    df["Soil_Type"] = df["Soil_Type_x"]

df = df.drop(columns=[c for c in df.columns if "Soil_Type_" in c])

# ---------------------------------------------------
# 2. Keep only ONE rainfall column
# ---------------------------------------------------

if "Rainfall_mm" in df.columns:
    df["Rainfall"] = df["Rainfall_mm"]

df = df.drop(columns=[c for c in ["Rainfall_mm"] if c in df.columns])

# ---------------------------------------------------
# 3. Drop unnecessary columns
# ---------------------------------------------------

drop_cols = ["Major_Crops"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# ---------------------------------------------------
# 4. Handle missing values
# ---------------------------------------------------

# Numeric → fill with median
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Categorical → fill with "Unknown"
categorical_cols = df.select_dtypes(include=["object"]).columns
df[categorical_cols] = df[categorical_cols].fillna("Unknown")

# ---------------------------------------------------
# 5. Label Encoding (instead of one-hot)
# ---------------------------------------------------

le = LabelEncoder()

for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# ---------------------------------------------------
# 6. Remove duplicates
# ---------------------------------------------------

df = df.drop_duplicates()

print("Cleaned shape:", df.shape)

# ---------------------------------------------------
# 7. Save cleaned dataset
# ---------------------------------------------------

df.to_csv("data/clean_master_dataset_v2.csv", index=False)

print("Clean dataset saved successfully!")