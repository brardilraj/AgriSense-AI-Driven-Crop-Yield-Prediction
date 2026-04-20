import pandas as pd

# -------------------------------
# STEP 1: Load rainfall dataset
# -------------------------------
rain_df = pd.read_csv(
    "data/rainfall_data.csv",
    skiprows=4
)

print("Raw rainfall data:")
print(rain_df.head())
print(rain_df.columns)

# --------------------------------
# STEP 2: Rename required columns
# --------------------------------
rain_df = rain_df.rename(columns={
    "Unnamed: 1": "District",
    "Actual": "June_Actual",
    "Actual.1": "July_Actual",
    "Actual.2": "August_Actual"
})

# --------------------------------
# STEP 3: Keep only required columns
# --------------------------------
rain_df = rain_df[["District", "June_Actual", "July_Actual", "August_Actual"]]

# --------------------------------
# STEP 4: Remove invalid rows
# --------------------------------

# Drop rows where District is missing
rain_df = rain_df.dropna(subset=["District"])

# Remove rows where District is numeric (e.g., 1, 2, 3)
rain_df = rain_df[~rain_df["District"].astype(str).str.isnumeric()]

# Remove header-like rows
rain_df = rain_df[rain_df["District"].str.lower() != "district"]

# --------------------------------
# STEP 5: Convert rainfall values to numeric
# --------------------------------
for col in ["June_Actual", "July_Actual", "August_Actual"]:
    rain_df[col] = pd.to_numeric(rain_df[col], errors="coerce")

# --------------------------------
# STEP 6: Calculate average monsoon rainfall
# --------------------------------
rain_df["Rainfall"] = rain_df[
    ["June_Actual", "July_Actual", "August_Actual"]
].mean(axis=1)

# --------------------------------
# STEP 7: Add year column
# --------------------------------
rain_df["Year"] = 2023   # Change if dataset year is different

# --------------------------------
# STEP 8: Final column selection
# --------------------------------
rain_df = rain_df[["District", "Year", "Rainfall"]]

print("\nCleaned rainfall data:")
print(rain_df.head())

# --------------------------------
# STEP 9: Save cleaned rainfall dataset
# --------------------------------
rain_df.to_csv("data/clean_rainfall.csv", index=False)

print("\nClean rainfall dataset saved successfully!")
