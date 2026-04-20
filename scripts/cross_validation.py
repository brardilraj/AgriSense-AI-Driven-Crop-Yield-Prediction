import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("data/clean_master_dataset_v2.csv")

# Define target
y = df["Yield"]

# Define features
X = df.drop(columns=["Yield"])

# Create model
model = RandomForestRegressor(n_estimators=300, random_state=42)

# Perform 5-fold cross validation
scores = cross_val_score(model, X, y, cv=5, scoring="r2")

print("Fold R² Scores:", scores)
print("Average R² Score:", scores.mean())
print("Standard Deviation:", scores.std())