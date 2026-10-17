
from flask import Flask, jsonify, send_from_directory, request
import requests
import csv
import os
from datetime import datetime, timedelta

app = Flask(__name__)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "weather_data.csv")

# Use your ESP32's current IP address
ESP32_URL = "http://10.163.181.58/api/data"


def read_number(value):
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except (ValueError, TypeError):
        return None


@app.route("/")
def dashboard():
    return send_from_directory(BASE_DIR, "dashboard.html")


@app.route("/api/weather")
def weather():
    try:
        response = requests.get(ESP32_URL, timeout=3)
        response.raise_for_status()
        data = response.json()

        def get_value(*names):
            for name in names:
                if data.get(name) is not None:
                    return read_number(data[name])
            return None

        return jsonify({
            "temperature": get_value("temperature", "temp", "Temperature"),
            "humidity": get_value("humidity", "Humidity"),
            "pressure": get_value("pressure", "Pressure")
        })

    except (requests.RequestException, ValueError) as error:
        return jsonify({
            "error": "Could not read the ESP32 sensor API",
            "details": str(error)
        }), 502


@app.route("/api/history")
def history():
    selected_range = request.args.get("range", "24h")
    now = datetime.now()

    ranges = {
        "1h": timedelta(hours=1),
        "6h": timedelta(hours=6),
        "24h": timedelta(hours=24)
    }

    cutoff = now - ranges[selected_range] if selected_range in ranges else None
    records = []

    if not os.path.exists(CSV_FILE):
        return jsonify({
            "records": [],
            "message": "weather_data.csv not found. Start data_logger.py."
        })

    try:
        with open(CSV_FILE, "r", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)

            for row in reader:
                timestamp_text = row.get("timestamp", "")

                try:
                    timestamp = datetime.fromisoformat(
                        timestamp_text.strip().replace("Z", "+00:00")
                    )
                    # Convert timezone-aware timestamps to local naive time
                    if timestamp.tzinfo is not None:
                        timestamp = timestamp.astimezone().replace(tzinfo=None)
                except (ValueError, AttributeError):
                    continue

                if cutoff is not None and timestamp < cutoff:
                    continue

                records.append({
                    "timestamp": timestamp.isoformat(timespec="seconds"),
                    "temperature": read_number(row.get("temperature")),
                    "humidity": read_number(row.get("humidity")),
                    "pressure": read_number(row.get("pressure"))
                })

    except (OSError, csv.Error) as error:
        return jsonify({"error": str(error)}), 500

    records.sort(key=lambda item: item["timestamp"])

    return jsonify({
        "records": records,
        "count": len(records),
        "range": selected_range
    })


@app.route("/api/online-weather")
def online_weather():
    latitude = 10.5276
    longitude = 76.2144

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": "Asia/Kolkata",
        "current": (
            "temperature_2m,relative_humidity_2m,"
            "precipitation,cloud_cover,wind_speed_10m,"
            "wind_direction_10m,wind_gusts_10m"
        ),
        "hourly": (
            "temperature_2m,precipitation_probability,"
            "precipitation,uv_index,cloud_cover,visibility,"
            "wind_speed_10m,wind_direction_10m"
        ),
        "daily": (
            "sunrise,sunset,daylight_duration,"
            "temperature_2m_max,temperature_2m_min,"
            "precipitation_sum,precipitation_probability_max,"
            "wind_speed_10m_max,uv_index_max"
        ),
        "forecast_days": 7
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return jsonify(response.json())
    except requests.RequestException as error:
        return jsonify({"error": str(error)}), 502

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)




