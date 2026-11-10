
import pandas as pd
import sys

# Accept horizon minutes as optional argument: python prepare_data.py 10
# Default is 10 minutes if not specified.
HORIZONS = [10, 20, 30]

if len(sys.argv) > 1:
    try:
        HORIZONS = [int(sys.argv[1])]
    except ValueError:
        pass

# Load original ESP32 sensor data
df = pd.read_csv("weather_data.csv")

required_columns = ["timestamp", "temperature", "humidity", "pressure"]
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError(f"Missing columns in weather_data.csv: {missing_columns}")

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

for col in ["temperature", "humidity", "pressure"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

for horizon in HORIZONS:
    print(f"\n--- Preparing {horizon}-minute training data ---")

    future = df[["timestamp", "temperature", "humidity", "pressure"]].copy()
    future = future.rename(columns={
        "timestamp":   "future_timestamp",
        "temperature": "future_temperature",
        "humidity":    "future_humidity",
        "pressure":    "future_pressure"
    })

    current = df.copy()
    current["target_timestamp"] = current["timestamp"] + pd.Timedelta(minutes=horizon)

    data = pd.merge_asof(
        current.sort_values("target_timestamp"),
        future.sort_values("future_timestamp"),
        left_on="target_timestamp",
        right_on="future_timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=90),
        allow_exact_matches=True
    )

    features = ["temperature", "humidity", "pressure"]
    targets  = ["future_temperature", "future_humidity", "future_pressure"]

    data = data.dropna(subset=features + targets).reset_index(drop=True)
    data = data.drop(columns=["target_timestamp"])

    out_file = f"weather_training_{horizon}min.csv"
    data.to_csv(out_file, index=False)
    if horizon == 10:
        data.to_csv("weather_training_data.csv", index=False)

    print(f"Saved: {out_file}")
    print(f"Training rows: {len(data)}")
    print(f"Prediction horizon: {horizon} minutes")
    print(data[["timestamp", "temperature", "humidity", "pressure",
                "future_timestamp", "future_temperature",
                "future_humidity", "future_pressure"]].head(3))

print("\nAll training files prepared successfully!")
