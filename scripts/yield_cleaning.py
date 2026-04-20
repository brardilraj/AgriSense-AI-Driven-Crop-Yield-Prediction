import pandas as pd

# Load dataset
df = pd.read_csv("data/Tamilnadu agriculture yield data.csv")

# Check first few rows
print(df.head())

# Check column names
print(df.columns)
# Keep only Tamil Nadu data
df = df[df["State_Name"] == "Tamil Nadu"]

print("After filtering Tamil Nadu:")
print(df.shape)
# Drop rows with missing Area or Production
df = df.dropna(subset=["Area", "Production"])

print("After removing missing values:")
print(df.shape)
# Calculate yield
df["Yield"] = df["Production"] / df["Area"]
# Keep only useful columns
df = df[[
    "District_Name",
    "Crop_Year",
    "Season",
    "Crop",
    "Area",
    "Yield"
]]

print(df.head())
# Save cleaned dataset
df.to_csv("data/clean_yield.csv", index=False)

print("Clean yield dataset saved successfully!")
