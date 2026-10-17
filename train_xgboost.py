import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# Load prepared data
df = pd.read_csv("weather_training_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Input features: current temperature, humidity, and time of day
minutes = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute

df["time_sin"] = np.sin(2 * np.pi * minutes / 1440)
df["time_cos"] = np.cos(2 * np.pi * minutes / 1440)

features = ["temperature", "humidity", "time_sin", "time_cos"]
targets = ["future_temperature", "future_humidity"]

df = df.dropna(subset=features + targets).reset_index(drop=True)

if len(df) < 20:
    raise ValueError(
        f"Only {len(df)} training rows. Collect more data before training."
    )

# Keep time order: first 80% for training, last 20% for testing
split = int(len(df) * 0.8)

X_train = df[features].iloc[:split]
X_test = df[features].iloc[split:]
y_train = df[targets].iloc[:split]
y_test = df[targets].iloc[split:]

model = MultiOutputRegressor(
    XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        objective="reg:squarederror",
        n_jobs=-1
    )
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

for i, name in enumerate(["Temperature", "Humidity"]):
    actual = y_test.iloc[:, i]
    predicted = predictions[:, i]

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    # Simple baseline: assume the current reading stays unchanged
    current = (
        df["temperature"].iloc[split:].to_numpy()
        if i == 0
        else df["humidity"].iloc[split:].to_numpy()
    )
    baseline_mae = mean_absolute_error(actual, current)

    print(f"\n{name} prediction (10 minutes ahead)")
    print(f"Model MAE: {mae:.3f}")
    print(f"Model RMSE: {rmse:.3f}")
    print(f"Baseline MAE: {baseline_mae:.3f}")

# Save the trained models
import joblib
joblib.dump(model, "xgboost_weather_model.pkl")
print("\nModel saved as xgboost_weather_model.pkl")

