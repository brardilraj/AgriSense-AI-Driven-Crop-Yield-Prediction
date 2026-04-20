import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load final dataset
df = pd.read_csv("data/final_dataset.csv")

print("Original dataset:")
print(df.head())

# -----------------------------
# Encode categorical variables
# -----------------------------
encoder = LabelEncoder()

for col in ["District_Name", "Season", "Crop"]:
    df[col] = encoder.fit_transform(df[col])

# -----------------------------
# Separate features and target
# -----------------------------
X = df.drop("Yield", axis=1)
y = df["Yield"]

# -----------------------------
# Scale numerical features
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nPreprocessing completed!")
print("Feature matrix shape:", X_scaled.shape)
print("Target shape:", y.shape)
