import pandas as pd

df = pd.read_csv("weather_clean.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])
df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
df["humidity"] = pd.to_numeric(df["humidity"], errors="coerce")

df = df.dropna(subset=["timestamp", "temperature", "humidity"])
df = df.sort_values("timestamp").reset_index(drop=True)

future = df[["timestamp", "temperature", "humidity"]].copy()
future["timestamp"] = future["timestamp"] - pd.Timedelta(minutes=10)

future = future.rename(columns={
    "temperature": "future_temperature",
    "humidity": "future_humidity"
})

data = pd.merge_asof(
    df,
    future.sort_values("timestamp"),
    on="timestamp",
    direction="nearest",
    tolerance=pd.Timedelta(seconds=30)
)

data = data.dropna(
    subset=["future_temperature", "future_humidity"]
)

data.to_csv("weather_training_data.csv", index=False)

print("Prepared data saved successfully!")
print("Training rows:", len(data))
print(data.head())
