
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

HORIZONS = [10, 20, 30]

for horizon in HORIZONS:
    print(f"\n{'='*50}")
    print(f"Training LSTM — {horizon}-minute forecast")
    print(f"{'='*50}")

    in_file = f"weather_training_{horizon}min.csv"
    df = pd.read_csv(in_file)

    features = ["temperature", "humidity", "pressure"]
    targets  = ["future_temperature", "future_humidity", "future_pressure"]

    df = df.dropna(subset=features + targets).reset_index(drop=True)

    if len(df) < 50:
        raise ValueError(f"Only {len(df)} rows for {horizon}min LSTM. Collect more data.")

    split = int(len(df) * 0.8)

    X_raw = df[features].to_numpy(dtype=float)
    y_raw = df[targets].to_numpy(dtype=float)

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    x_scaler.fit(X_raw[:split])
    y_scaler.fit(y_raw[:split])

    X_scaled = x_scaler.transform(X_raw)
    y_scaled = y_scaler.transform(y_raw)

    window = 10
    X, y, target_indices = [], [], []

    for i in range(window - 1, len(df)):
        X.append(X_scaled[i - window + 1:i + 1])
        y.append(y_scaled[i])
        target_indices.append(i)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    target_indices = np.array(target_indices)

    train_mask = target_indices < split
    test_mask  = target_indices >= split

    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Not enough rows to build train/test windows.")

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(window, len(features))),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(3)
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(
        X_train, y_train,
        epochs=40,
        batch_size=16,
        validation_split=0.1,
        shuffle=False,
        verbose=1
    )

    pred_scaled = model.predict(X_test, verbose=0)
    pred   = y_scaler.inverse_transform(pred_scaled)
    actual = y_scaler.inverse_transform(y_test)

    for i, name in enumerate(["Temperature", "Humidity", "Pressure"]):
        mae  = mean_absolute_error(actual[:, i], pred[:, i])
        rmse = np.sqrt(mean_squared_error(actual[:, i], pred[:, i]))
        unit = "°C" if i == 0 else "%" if i == 1 else "hPa"
        print(f"  {name}: MAE={mae:.3f}{unit}  RMSE={rmse:.3f}{unit}")

    model.save(f"lstm_{horizon}min.keras")
    joblib.dump(x_scaler, f"lstm_x_scaler_{horizon}min.pkl")
    joblib.dump(y_scaler, f"lstm_y_scaler_{horizon}min.pkl")
    print(f"Saved: lstm_{horizon}min.keras + scalers")

print("\nAll LSTM models trained successfully!")
