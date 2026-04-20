import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import xgboost as xgb

# DNN libraries
from tensorflow import keras
from scikeras.wrappers import KerasRegressor


# =========================
# Load dataset
# =========================

df = pd.read_csv("data/clean_master_dataset_v2.csv")

# Target
y = df["Yield"]

# Features
X = df.drop(columns=["Yield"])


# =========================
# Define DNN model
# =========================

def build_dnn():

    model = keras.Sequential([
        keras.layers.Dense(128, activation="relu", input_shape=(X.shape[1],)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model


# Wrap DNN for sklearn
dnn_model = Pipeline([
    ("scaler", StandardScaler()),
    ("dnn", KerasRegressor(
        model=build_dnn,
        epochs=50,
        batch_size=32,
        verbose=0
    ))
])


# =========================
# Define models
# =========================

models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        random_state=42
    ),

    "XGBoost": xgb.XGBRegressor(
        n_estimators=350,
        learning_rate=0.05,
        random_state=42
    ),

    "Deep Neural Network": dnn_model
}


# =========================
# Cross validation setup
# =========================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


print("\n===== Model Comparison Using 5-Fold Cross Validation =====\n")


# =========================
# Run comparison
# =========================

results = {}

for name, model in models.items():

    print(f"Running {name}...")

    scores = cross_val_score(
        model,
        X,
        y,
        cv=kf,
        scoring="r2"
    )

    results[name] = scores.mean()

    print("Fold Scores:", scores)
    print("Average R²:", scores.mean())
    print("Std Dev:", scores.std())
    print("----------------------------------")


# =========================
# Final summary
# =========================

print("\n===== Final Model Ranking =====\n")

for model, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{model} : {score:.3f}")