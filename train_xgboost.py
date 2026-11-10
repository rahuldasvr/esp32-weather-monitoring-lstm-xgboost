
import pandas as pd
import numpy as np
import joblib

from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

HORIZONS = [10, 20, 30]

for horizon in HORIZONS:
    print(f"\n{'='*50}")
    print(f"Training XGBoost — {horizon}-minute forecast")
    print(f"{'='*50}")

    in_file = f"weather_training_{horizon}min.csv"
    df = pd.read_csv(in_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Time-of-day features
    mins = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    df["time_sin"] = np.sin(2 * np.pi * mins / 1440)
    df["time_cos"] = np.cos(2 * np.pi * mins / 1440)

    features = ["temperature", "humidity", "pressure", "time_sin", "time_cos"]
    targets  = ["future_temperature", "future_humidity", "future_pressure"]

    df = df.dropna(subset=features + targets).reset_index(drop=True)

    if len(df) < 20:
        raise ValueError(f"Only {len(df)} rows for {horizon}min model. Collect more data.")

    split = int(len(df) * 0.8)
    X_train = df[features].iloc[:split]
    X_test  = df[features].iloc[split:]
    y_train = df[targets].iloc[:split]
    y_test  = df[targets].iloc[split:]

    model = MultiOutputRegressor(
        XGBRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            objective="reg:squarederror",
            n_jobs=-1
        )
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    for i, name in enumerate(["Temperature", "Humidity", "Pressure"]):
        actual    = y_test.iloc[:, i].to_numpy()
        predicted = predictions[:, i]
        mae  = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        unit = "°C" if i == 0 else "%" if i == 1 else "hPa"
        print(f"  {name}: MAE={mae:.3f}{unit}  RMSE={rmse:.3f}{unit}")

    out_model    = f"xgboost_{horizon}min.pkl"
    out_features = f"xgboost_features_{horizon}min.pkl"
    joblib.dump(model,    out_model)
    joblib.dump(features, out_features)
    print(f"Saved: {out_model}, {out_features}")

print("\nAll XGBoost models trained successfully!")
