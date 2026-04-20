import pandas as pd

# Load cleaned dataset
df = pd.read_csv("data/clean_master_dataset_v2.csv")

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns)

# -----------------------------
# Define target
# -----------------------------
target = "Yield"

# -----------------------------
# Define features
# -----------------------------
X = df.drop(columns=[target])
y = df[target]

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)

print("\nFirst few target values:")
print(y.head())