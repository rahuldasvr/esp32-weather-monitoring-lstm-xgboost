
import pandas as pd
from pathlib import Path

input_file = Path("weather_data.csv")
output_file = Path("weather_clean.csv")

print("Starting weather data cleaning...")

if not input_file.exists():
    print("ERROR: weather_data.csv not found!")
    raise SystemExit

df = pd.read_csv(input_file)

print("Original rows:", len(df))
print("Columns:", list(df.columns))

required = ["timestamp", "temperature", "humidity"]

if not all(col in df.columns for col in required):
    print("ERROR: Required columns are missing.")
    raise SystemExit

df["timestamp"] = pd.to_datetime(
    df["timestamp"], errors="coerce"
)

df["temperature"] = pd.to_numeric(
    df["temperature"], errors="coerce"
)

df["humidity"] = pd.to_numeric(
    df["humidity"], errors="coerce"
)

# Remove rows without valid timestamps or readings
df = df.dropna(
    subset=["timestamp", "temperature", "humidity"]
)

# Remove duplicate timestamps
df = df.drop_duplicates(
    subset=["timestamp"], keep="last"
)

# Sort readings by time
df = df.sort_values("timestamp")

# Basic sensor-range checks
df = df[
    df["temperature"].between(-40, 85)
    & df["humidity"].between(0, 100)
]

df.to_csv(output_file, index=False)

print("Cleaning completed!")
print("Clean rows:", len(df))
print("Created:", output_file)

print(df.head())
