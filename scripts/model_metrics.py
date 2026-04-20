import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import xgboost as xgb

# DNN
from tensorflow import keras
from scikeras.wrappers import KerasRegressor


# =========================
# Load Dataset
# =========================

df = pd.read_csv("data/clean_master_dataset_v2.csv")

y = df["Yield"]
X = df.drop(columns=["Yield"])


# =========================
# DNN Model
# =========================

def build_dnn():

    model = keras.Sequential([
        keras.layers.Input(shape=(X.shape[1],)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model


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
# Define Models
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
# Cross Validation
# =========================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("\n===== Cross Validation Performance Metrics =====\n")


# =========================
# Run Metrics
# =========================

results = {}

for name, model in models.items():

    print(f"Running {name}...")

    scores = cross_validate(
        model,
        X,
        y,
        cv=kf,
        scoring={
            "r2": "r2",
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error"
        }
    )

    r2 = scores["test_r2"].mean()
    mae = -scores["test_mae"].mean()
    rmse = -scores["test_rmse"].mean()

    print("Average R²:", r2)
    print("Average MAE:", mae)
    print("Average RMSE:", rmse)
    print("----------------------------")

    results[name] = (r2, mae, rmse)


# =========================
# Final Summary Table
# =========================

print("\n===== Final Model Performance =====\n")

print("Model\t\tR²\t\tMAE\t\tRMSE")

for model, vals in results.items():
    print(f"{model:20s} {vals[0]:.3f} \t {vals[1]:.2f} \t {vals[2]:.2f}")