
import requests
import csv
import os
import time
from datetime import datetime

# Your ESP32 weather station
ESP32_URL = "http://10.163.181.58/api/data"

# CSV file created in your project folder
CSV_FILE = "weather_data.csv"

print("ESP32 Weather Data Logger Started!")
print("Collecting temperature and humidity...")
print("Press Ctrl+C to stop.\n")

try:
    while True:
        try:
            # Request data from ESP32
            response = requests.get(ESP32_URL, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Read sensor values
            temperature = data.get("temperature")
            humidity = data.get("humidity")
            pressure = data.get("pressure")

            # Temperature and humidity are required.
            # Pressure is optional.
            if temperature is None or humidity is None:
                print("Temperature or humidity unavailable:", data)

            else:
                # Leave pressure blank if it is unavailable
                if pressure is None:
                    pressure = ""

                # Add a header only if the file is new or empty
                file_exists = (
                    os.path.exists(CSV_FILE)
                    and os.path.getsize(CSV_FILE) > 0
                )

                with open(CSV_FILE, "a", newline="",
                          encoding="utf-8") as file:
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

                print(
                    f"Saved | Temperature: {temperature} °C | "
                    f"Humidity: {humidity}% | "
                    f"Pressure: {pressure if pressure != '' else 'Unavailable'}"
                )

        except requests.exceptions.RequestException as error:
            print("Connection error:", error)

        except (ValueError, TypeError) as error:
            print("Data reading error:", error)

        time.sleep(30)

except KeyboardInterrupt:
    print("\nData collection stopped.")
