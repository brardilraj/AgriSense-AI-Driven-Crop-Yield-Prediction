import pandas as pd

# -------------------------------
# STEP 1: Load cleaned datasets
# -------------------------------
yield_df = pd.read_csv("data/clean_yield.csv")
rain_df = pd.read_csv("data/clean_rainfall.csv")

print("Yield dataset preview:")
print(yield_df.head())
print(yield_df.columns)

print("\nRainfall dataset preview:")
print(rain_df.head())
print(rain_df.columns)

# --------------------------------
# STEP 2: Standardize district names
# --------------------------------
yield_df["District_Name"] = (
    yield_df["District_Name"]
    .str.lower()
    .str.strip()
)

rain_df["District"] = (
    rain_df["District"]
    .str.lower()
    .str.strip()
)

# --------------------------------
# STEP 3: Merge datasets (DISTRICT ONLY)
# --------------------------------
merged_df = pd.merge(
    yield_df,
    rain_df,
    left_on="District_Name",
    right_on="District",
    how="inner"
)

print("\nMerged dataset preview:")
print(merged_df.head())
print("Merged shape:", merged_df.shape)

# --------------------------------
# STEP 4: Drop duplicate / unused columns
# --------------------------------
merged_df.drop(columns=["District", "Year"], inplace=True)

# --------------------------------
# STEP 5: Final column selection
# --------------------------------
merged_df = merged_df[
    ["District_Name", "Crop_Year", "Season", "Crop", "Area", "Yield", "Rainfall"]
]

print("\nFinal merged dataset:")
print(merged_df.head())

# --------------------------------
# STEP 6: Save final dataset
# --------------------------------
merged_df.to_csv("data/final_dataset.csv", index=False)

print("\nFinal merged dataset saved successfully!")
