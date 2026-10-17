import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load prepared data
df = pd.read_csv("weather_training_data.csv")

features = ["temperature", "humidity"]
targets = ["future_temperature", "future_humidity"]

df = df.dropna(subset=features + targets).reset_index(drop=True)

if len(df) < 50:
    raise ValueError(
        f"Only {len(df)} rows available. Collect more data before training LSTM."
    )

# Keep chronological order: first 80% train, last 20% test
split = int(len(df) * 0.8)

X_raw = df[features].to_numpy()
y_raw = df[targets].to_numpy()

# Fit scalers using training data only
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

x_scaler.fit(X_raw[:split])
y_scaler.fit(y_raw[:split])

X_scaled = x_scaler.transform(X_raw)
y_scaled = y_scaler.transform(y_raw)

# Each sample uses the previous 10 readings
window = 10

X, y, target_indices = [], [], []

for i in range(window - 1, len(df)):
    X.append(X_scaled[i - window + 1:i + 1])
    y.append(y_scaled[i])
    target_indices.append(i)

X = np.array(X)
y = np.array(y)
target_indices = np.array(target_indices)

train_mask = target_indices < split
test_mask = target_indices >= split

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]

# Build the LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(window, len(features))),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(2)
])

model.compile(optimizer="adam", loss="mse")

model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=16,
    validation_split=0.1,
    shuffle=False,
    verbose=1
)

# Predict and convert back to actual units
pred_scaled = model.predict(X_test, verbose=0)
pred = y_scaler.inverse_transform(pred_scaled)
actual = y_scaler.inverse_transform(y_test)

for i, name in enumerate(["Temperature", "Humidity"]):
    mae = mean_absolute_error(actual[:, i], pred[:, i])
    rmse = np.sqrt(mean_squared_error(actual[:, i], pred[:, i]))
    print(f"\n{name} prediction 10 minutes ahead")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")

# Save model and scalers
model.save("lstm_weather_model.keras")

import joblib
joblib.dump(x_scaler, "lstm_x_scaler.pkl")
joblib.dump(y_scaler, "lstm_y_scaler.pkl")

print("\nLSTM model saved successfully!")
