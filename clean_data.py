
import pandas as pd
from pathlib import Path

input_file = Path("weather_data.csv")
output_file = Path("weather_clean.csv")

print("Starting weather data cleaning...")

if not input_file.exists():
    raise FileNotFoundError("ERROR: weather_data.csv not found!")

df = pd.read_csv(input_file)

print("Original rows:", len(df))
print("Columns:", list(df.columns))

# Required columns
required = ["timestamp", "temperature", "humidity", "pressure"]

missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(
        f"Required columns are missing: {missing}"
    )

# Convert timestamp
df["timestamp"] = pd.to_datetime(
    df["timestamp"], errors="coerce"
)

# Convert sensor values to numeric
for col in ["temperature", "humidity", "pressure"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove rows without valid timestamps, temperature, or humidity.
# Pressure is optional in the raw dataset.
df = df.dropna(
    subset=["timestamp", "temperature", "humidity"]
)

# Remove duplicate timestamps
df = df.drop_duplicates(
    subset=["timestamp"], keep="last"
)

# Sort chronologically
df = df.sort_values("timestamp").reset_index(drop=True)

# Basic sensor-range checks
df = df[
    df["temperature"].between(-40, 85)
    & df["humidity"].between(0, 100)
]

# Keep pressure only when within a broad plausible range.
# Invalid or missing pressure becomes NaN, not zero.
invalid_pressure = (
    df["pressure"].notna()
    & ~df["pressure"].between(300, 1100)
)

df.loc[invalid_pressure, "pressure"] = float("nan")

# Save cleaned data; original CSV remains unchanged
df.to_csv(output_file, index=False)

print("\nCleaning completed!")
print("Clean rows:", len(df))
print("Rows with valid pressure:", df["pressure"].notna().sum())
print("Rows missing pressure:", df["pressure"].isna().sum())
print("Created:", output_file)

print(df.head())
