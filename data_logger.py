
import requests
import csv
import os
import time
from datetime import datetime

# Current ESP32 weather station IP
ESP32_URL = "http://10.55.192.58/api/data"

# CSV file in your project folder
CSV_FILE = "weather_data.csv"

print("ESP32 Weather Data Logger Started!")
print("Collecting temperature, humidity, and pressure...")
print("Logging interval: 30 seconds")
print("Press Ctrl+C to stop.\n")

try:
    while True:
        try:
            # Request data from ESP32
            response = requests.get(ESP32_URL, timeout=10)
            response.raise_for_status()
            data = response.json()

            temperature = data.get("temperature")
            humidity = data.get("humidity")
            pressure = data.get("pressure")

            # Temperature and humidity are required
            if temperature is None or humidity is None:
                print("Temperature or humidity unavailable:", data)

            else:
                # Keep missing pressure blank, not zero
                if pressure is None:
                    pressure = ""

                # Add header only if file is new or empty
                file_exists = (
                    os.path.exists(CSV_FILE)
                    and os.path.getsize(CSV_FILE) > 0
                )

                with open(
                    CSV_FILE,
                    "a",
                    newline="",
                    encoding="utf-8"
                ) as file:
                    writer = csv.writer(file)

                    if not file_exists:
                        writer.writerow([
                            "timestamp",
                            "temperature",
                            "humidity",
                            "pressure"
                        ])

                    writer.writerow([
                        datetime.now().isoformat(timespec="seconds"),
                        temperature,
                        humidity,
                        pressure
                    ])

                pressure_display = (
                    f"{pressure} hPa"
                    if pressure != ""
                    else "Unavailable"
                )

                print(
                    f"Saved | Temperature: {temperature} °C | "
                    f"Humidity: {humidity}% | "
                    f"Pressure: {pressure_display}"
                )

        except requests.exceptions.RequestException as error:
            print("Connection error:", error)

        except (ValueError, TypeError) as error:
            print("Data reading error:", error)

        time.sleep(30)

except KeyboardInterrupt:
    print("\nData collection stopped.")
