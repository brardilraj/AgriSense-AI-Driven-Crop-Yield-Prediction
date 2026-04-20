import pandas as pd
from sklearn.model_selection import train_test_split

# Load cleaned dataset
df = pd.read_csv("data/clean_master_dataset_v2.csv")

# -----------------------------
# Define target and features
# -----------------------------
target = "Yield"

X = df.drop(columns=[target])
y = df[target]

# -----------------------------
# Perform Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% test data
    random_state=42     # ensures reproducibility
)

# -----------------------------
# Display results
# -----------------------------
print("Total samples:", len(df))
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nFeature shape (train):", X_train.shape)
print("Feature shape (test):", X_test.shape)

print("\nTarget shape (train):", y_train.shape)
print("Target shape (test):", y_test.shape)