import pandas as pd

# Load datasets
final_df = pd.read_csv("data/final_dataset.csv")
soil_df = pd.read_csv("data/soil_dataset.csv")
agri_df = pd.read_csv("data/agri_dataset.csv")

# Standardize district names
final_df["District_Name"] = final_df["District_Name"].str.lower().str.strip()
soil_df["District"] = soil_df["District"].str.lower().str.strip()
agri_df["District"] = agri_df["District"].str.lower().str.strip()

# Merge final dataset with soil dataset
merged_df = pd.merge(
    final_df,
    soil_df,
    left_on="District_Name",
    right_on="District",
    how="left"
)

# Merge with agriculture dataset
merged_df = pd.merge(
    merged_df,
    agri_df,
    left_on="District_Name",
    right_on="District",
    how="left"
)

# Drop duplicate district columns
merged_df = merged_df.drop(columns=["District_x", "District_y"])

# Save master dataset
merged_df.to_csv("data/master_dataset.csv", index=False)

print("Master dataset created successfully!")
print(merged_df.head())