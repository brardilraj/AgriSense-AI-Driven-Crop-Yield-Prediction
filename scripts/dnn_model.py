import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("data/clean_master_dataset_v2.csv")

target = "Yield"

X = df.drop(columns=[target])
y = df[target]

# -----------------------------
# Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# IMPORTANT: Scale features
# -----------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# Build Neural Network
# -----------------------------
model = Sequential()

model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))

# Output layer (regression)
model.add(Dense(1))

model.compile(
    optimizer='adam',
    loss='mse'
)

# -----------------------------
# Train model
# -----------------------------
model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# -----------------------------
# Predictions
# -----------------------------
y_pred = model.predict(X_test).flatten()

# -----------------------------
# Evaluation
# -----------------------------
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("DNN Results")
print("R² Score:", r2)
print("MAE:", mae)
print("RMSE:", rmse)