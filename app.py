from flask import Flask, jsonify, send_from_directory, request
import requests
import numpy as np
import joblib
import tensorflow as tf
import re
import csv
import pandas as pd
import os
import time
import threading
import secrets
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("[Serial] pyserial not installed. Serial reading disabled.")

from datetime import datetime, timedelta

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- Multi-horizon model registry: 10, 20, 30 minutes ----
HORIZONS = [10, 20, 30]

xgb_models      = {}
lstm_models     = {}
lstm_x_scalers  = {}
lstm_y_scalers  = {}

for _h in HORIZONS:
    _xgb_path  = os.path.join(BASE_DIR, f"xgboost_{_h}min.pkl")
    _lstm_path = os.path.join(BASE_DIR, f"lstm_{_h}min.keras")
    _lxs_path  = os.path.join(BASE_DIR, f"lstm_x_scaler_{_h}min.pkl")
    _lys_path  = os.path.join(BASE_DIR, f"lstm_y_scaler_{_h}min.pkl")

    if os.path.exists(_xgb_path):
        xgb_models[_h] = joblib.load(_xgb_path)
        print(f"[Models] XGBoost {_h}min loaded")
    else:
        print(f"[Models] WARNING: {_xgb_path} not found — run train_xgboost.py first")

    if os.path.exists(_lstm_path) and os.path.exists(_lxs_path) and os.path.exists(_lys_path):
        lstm_models[_h]    = tf.keras.models.load_model(_lstm_path)
        lstm_x_scalers[_h] = joblib.load(_lxs_path)
        lstm_y_scalers[_h] = joblib.load(_lys_path)
        print(f"[Models] LSTM {_h}min loaded")
    else:
        print(f"[Models] WARNING: LSTM {_h}min files missing — run train_lstm.py first")

CSV_FILE = os.path.join(BASE_DIR, "weather_data.csv")

# Change this IP if your ESP32 receives a different IP.
ESP32_URL = "http://10.55.192.58/api/data"
# Serial port config for direct USB reading
SERIAL_PORT = "COM5"
SERIAL_BAUD = 115200
SERIAL_TIMEOUT = 5

CLOUD_API_TOKEN = os.environ.get("CLOUD_API_TOKEN", "")

latest_cloud_reading = None
latest_cloud_reading_time = None

online_weather_cache = None
online_weather_cache_time = 0
ONLINE_WEATHER_CACHE_SECONDS = 600

city_weather_cache = None
city_weather_cache_time = 0
CITY_WEATHER_CACHE_SECONDS = 600

CITY_LOCATIONS = [
    {"city": "Kochi", "region": "Kerala", "group": "Kerala", "latitude": 9.9312, "longitude": 76.2673},
    {"city": "Thrissur", "region": "Kerala", "group": "Kerala", "latitude": 10.5276, "longitude": 76.2144},
    {"city": "Kozhikode", "region": "Kerala", "group": "Kerala", "latitude": 11.2588, "longitude": 75.7804},
    {"city": "Thiruvananthapuram", "region": "Kerala", "group": "Kerala", "latitude": 8.5241, "longitude": 76.9366},
    {"city": "Palakkad", "region": "Kerala", "group": "Kerala", "latitude": 10.7867, "longitude": 76.6548},
    {"city": "Delhi", "region": "India", "group": "India", "latitude": 28.6139, "longitude": 77.2090},
    {"city": "Mumbai", "region": "India", "group": "India", "latitude": 19.0760, "longitude": 72.8777},
    {"city": "Chennai", "region": "India", "group": "India", "latitude": 13.0827, "longitude": 80.2707},
    {"city": "Bengaluru", "region": "India", "group": "India", "latitude": 12.9716, "longitude": 77.5946},
    {"city": "Kolkata", "region": "India", "group": "India", "latitude": 22.5726, "longitude": 88.3639},
]


def read_number(value):
    try:
        if value is None or str(value).strip() == "":
            return None
        number = float(value)
        if not np.isfinite(number):
            return None
        return number
    except (ValueError, TypeError):
        return None


def running_on_render():
    return bool(os.environ.get("RENDER"))



def extract_sensor_reading(data):
    if not isinstance(data, dict):
        return None

    def get_value(*names):
        for name in names:
            if name in data and data[name] is not None:
                return read_number(data[name])
        return None

    return {
        "temperature": get_value("temperature", "temp", "Temperature"),
        "humidity": get_value("humidity", "hum", "Humidity"),
        "pressure": get_value("pressure", "pres", "Pressure"),
        "altitude": get_value("altitude", "Altitude"),
        "dhtConnected": data.get("dhtConnected", False),
        "bmpConnected": data.get("bmpConnected", False),
        "si1145Connected": data.get("si1145Connected", False),
        "visible": get_value("visible"),
        "ir": get_value("ir"),
        "uv": get_value("uv")
    }



@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-API-Token, Accept"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


latest_local_reading = None
latest_local_reading_time = None


# Thread/process lock for CSV writes — prevents duplicate writes from multiple threads/workers
_csv_lock = threading.Lock()

def safe_log_to_csv(reading):
    """Guarantees that NO duplicate or rapid-fire entries can ever be logged (< 25s apart).
    Uses a threading lock + re-checks the file inside the lock to prevent race conditions."""
    if not reading or reading.get("temperature") is None or reading.get("humidity") is None:
        return False
    now_dt = datetime.now()
    now_iso = now_dt.isoformat(timespec="seconds")

    with _csv_lock:
        # Re-read the last line INSIDE the lock to catch concurrent writers
        if os.path.exists(CSV_FILE) and os.path.getsize(CSV_FILE) > 0:
            try:
                with open(CSV_FILE, "r", newline="", encoding="utf-8-sig") as f:
                    lines = f.readlines()
                    for last_line in reversed(lines):
                        last_line = last_line.strip()
                        if not last_line or last_line.startswith("timestamp"):
                            continue
                        parts = last_line.split(",")
                        if parts and parts[0]:
                            try:
                                last_dt = datetime.fromisoformat(parts[0])
                                diff = (now_dt - last_dt).total_seconds()
                                if diff < 25:
                                    return False  # Too soon — reject duplicate
                            except ValueError:
                                pass
                        break
            except Exception:
                pass

        try:
            file_exists = os.path.exists(CSV_FILE) and os.path.getsize(CSV_FILE) > 0
            with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["timestamp", "temperature", "humidity", "pressure"])
                writer.writerow([
                    now_iso,
                    reading["temperature"],
                    reading["humidity"],
                    reading["pressure"] if reading.get("pressure") is not None else ""
                ])
            print(f"[Auto-Logger] Saved | {now_iso} | Temp: {reading['temperature']}C | Hum: {reading['humidity']}% | Pres: {reading.get('pressure')} hPa")
            return True
        except Exception as err:
            print(f"[Auto-Logger] Write error: {err}")
            return False


def try_read_serial():
    """Try to read one JSON line from ESP32 via USB serial (COM5)."""
    if not SERIAL_AVAILABLE:
        return None
    import json as _json
    try:
        with serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=SERIAL_TIMEOUT) as ser:
            deadline = time.time() + 12
            while time.time() < deadline:
                raw = ser.readline().decode("utf-8", errors="ignore").strip()
                if not raw:
                    continue
                # JSON line from ESP32
                if raw.startswith("{"):
                    try:
                        data = _json.loads(raw)
                        reading = extract_sensor_reading(data)
                        if reading and reading.get("temperature") is not None:
                            return reading
                    except Exception:
                        continue
                # Text line like: T:28.5 H:72.3 P:1013.2
                match = re.search(r'T:([\d.]+).*H:([\d.]+)(?:.*P:([\d.]+))?', raw, re.IGNORECASE)
                if match:
                    return {
                        "temperature": float(match.group(1)),
                        "humidity":    float(match.group(2)),
                        "pressure":    float(match.group(3)) if match.group(3) else None,
                        "altitude": None, "dhtConnected": True, "bmpConnected": False,
                        "si1145Connected": False, "visible": None, "ir": None, "uv": None
                    }
    except serial.SerialException as e:
        print(f"[Serial] Cannot open {SERIAL_PORT}: {e}")
    except Exception as e:
        print(f"[Serial] Error: {e}")
    return None


def try_read_http():
    """Try to read from ESP32 via HTTP."""
    try:
        res = requests.get(ESP32_URL, timeout=8)
        if res.status_code == 200:
            data = res.json()
            reading = extract_sensor_reading(data)
            if reading and reading.get("temperature") is not None:
                return reading
    except Exception as err:
        print(f"[HTTP] Cannot reach ESP32 at {ESP32_URL}: {type(err).__name__}")
    return None


def background_sensor_logger():
    """Polls ESP32 every 30s via Serial (primary) or HTTP (fallback) and logs to CSV."""
    global latest_local_reading, latest_local_reading_time
    print("[Background Logger] Started. Source: Serial (COM5) with HTTP fallback.")
    while True:
        if running_on_render():
            time.sleep(30)
            continue
        reading = None
        # Primary: USB Serial
        if SERIAL_AVAILABLE:
            reading = try_read_serial()
            if reading:
                print(f"[Serial] Got data  Temp={reading.get('temperature')}C  Hum={reading.get('humidity')}%")
        # Fallback: HTTP
        if reading is None:
            reading = try_read_http()
            if reading:
                print(f"[HTTP] Got data  Temp={reading.get('temperature')}C  Hum={reading.get('humidity')}%")
        # Log if we got a valid reading
        if reading:
            latest_local_reading = reading
            latest_local_reading_time = time.time()
            safe_log_to_csv(reading)
        else:
            print("[Background Logger] No data from Serial or HTTP. Retrying in 30s...")
        time.sleep(30)


# Start the background logger daemon thread
logger_thread = threading.Thread(target=background_sensor_logger, daemon=True)
logger_thread.start()


def get_latest_cached_reading():
    if not os.path.exists(CSV_FILE):
        return None
    try:
        with open(CSV_FILE, "r", newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
            if rows:
                last_row = rows[-1]
                return {
                    "temperature": read_number(last_row.get("temperature")),
                    "humidity": read_number(last_row.get("humidity")),
                    "pressure": read_number(last_row.get("pressure")),
                    "cached": True,
                    "timestamp": last_row.get("timestamp")
                }
    except Exception:
        pass
    return None


@app.route("/")
def dashboard():
    return send_from_directory(BASE_DIR, "dashboard.html")


@app.route("/api/weather")
def weather():
    if running_on_render():
        if latest_cloud_reading is None:
            return jsonify({
                "status": "error",
                "error": "Waiting for ESP32 cloud readings",
                "details": "No reading has been received yet."
            }), 503

        if latest_cloud_reading_time is None or time.time() - latest_cloud_reading_time > 120:
            return jsonify({
                "status": "error",
                "error": "ESP32 reading is stale",
                "details": "Waiting for a fresh sensor reading."
            }), 503

        return jsonify({"status": "success", **latest_cloud_reading})

    # Serve the fresh reading from the background logger if available (< 45s old)
    global latest_local_reading, latest_local_reading_time
    if latest_local_reading is not None and latest_local_reading_time is not None:
        if time.time() - latest_local_reading_time < 45:
            return jsonify({"status": "success", **latest_local_reading})

    try:
        response = requests.get(ESP32_URL, timeout=5)
        response.raise_for_status()
        data = response.json()
        reading = extract_sensor_reading(data)

        if reading is None:
            return jsonify({
                "status": "error",
                "error": "ESP32 returned an invalid JSON object"
            }), 502

        if reading["temperature"] is None and reading["humidity"] is None:
            return jsonify({
                "status": "error",
                "error": "Temperature and humidity are missing",
                "esp32_response": data
            }), 502

        latest_local_reading = reading
        latest_local_reading_time = time.time()
        return jsonify({"status": "success", **reading})

    except requests.exceptions.RequestException as error:
        fallback = get_latest_cached_reading()
        if fallback and fallback.get("temperature") is not None:
            return jsonify({
                "status": "success",
                **fallback,
                "warning": "ESP32 connection failed. Showing last logged reading."
            })
        return jsonify({
            "status": "error",
            "error": "Could not connect to the ESP32",
            "details": str(error)
        }), 502
    except ValueError as error:
        return jsonify({
            "status": "error",
            "error": "ESP32 returned invalid JSON",
            "details": str(error)
        }), 502


@app.route("/api/esp32-update", methods=["POST"])
def esp32_update():
    global latest_cloud_reading, latest_cloud_reading_time

    if not CLOUD_API_TOKEN:
        return jsonify({"error": "Cloud token is not configured"}), 503

    supplied_token = request.headers.get("X-API-Token", "")
    if not secrets.compare_digest(supplied_token, CLOUD_API_TOKEN):
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON"}), 400

    reading = extract_sensor_reading(data)
    if reading is None or reading["temperature"] is None or reading["humidity"] is None:
        return jsonify({"error": "Valid temperature and humidity are required"}), 400

    latest_cloud_reading = reading
    latest_cloud_reading_time = time.time()
    return jsonify({"status": "ok"}), 200
@app.route("/api/history")
def history():
    selected_range = request.args.get("range", "24h")
    ranges = {
        "1h": timedelta(hours=1),
        "6h": timedelta(hours=6),
        "24h": timedelta(hours=24)
    }
    cutoff = datetime.now() - ranges[selected_range] if selected_range in ranges else None
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
                try:
                    timestamp = datetime.fromisoformat(
                        row.get("timestamp", "").strip().replace("Z", "+00:00")
                    )
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
    global online_weather_cache, online_weather_cache_time

    if (
        online_weather_cache is not None
        and time.time() - online_weather_cache_time < ONLINE_WEATHER_CACHE_SECONDS
    ):
        result = jsonify(online_weather_cache)
        result.headers["X-Weather-Cache"] = "HIT"
        return result

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 10.5276,
        "longitude": 76.2144,
        "timezone": "Asia/Kolkata",
        "current": "temperature_2m,relative_humidity_2m,precipitation,cloud_cover,wind_speed_10m,wind_direction_10m,wind_gusts_10m",
        "hourly": "temperature_2m,precipitation_probability,precipitation,uv_index,cloud_cover,visibility,wind_speed_10m,wind_direction_10m",
        "daily": "sunrise,sunset,daylight_duration,temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,wind_speed_10m_max,uv_index_max",
        "forecast_days": 7
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        online_weather_cache = data
        online_weather_cache_time = time.time()

        result = jsonify(data)
        result.headers["X-Weather-Cache"] = "MISS"
        return result

    except requests.RequestException as error:
        if online_weather_cache is not None:
            result = jsonify(online_weather_cache)
            result.headers["X-Weather-Cache"] = "STALE"
            return result

        status_code = getattr(getattr(error, "response", None), "status_code", None)
        return jsonify({
            "error": "Online weather service is temporarily unavailable",
            "details": f"Open-Meteo returned HTTP {status_code}" if status_code else str(error)
        }), 502


@app.route("/api/city-weather")
def city_weather():
    global city_weather_cache, city_weather_cache_time

    if (
        city_weather_cache is not None
        and time.time() - city_weather_cache_time < CITY_WEATHER_CACHE_SECONDS
    ):
        result = jsonify({"cities": city_weather_cache})
        result.headers["X-Weather-Cache"] = "HIT"
        return result

    url = "https://api.open-meteo.com/v1/forecast"
    results = []
    errors = []

    for place in CITY_LOCATIONS:
        params = {
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "timezone": "Asia/Kolkata",
            "current": "temperature_2m,relative_humidity_2m"
        }

        try:
            response = requests.get(url, params=params, timeout=8)
            response.raise_for_status()
            payload = response.json()
            current = payload.get("current", {})

            results.append({
                "city": place["city"],
                "region": place["region"],
                "group": place["group"],
                "temperature": read_number(current.get("temperature_2m")),
                "humidity": read_number(current.get("relative_humidity_2m")),
                "time": current.get("time")
            })

        except (requests.RequestException, ValueError) as error:
            errors.append({"city": place["city"], "error": str(error)})
            results.append({
                "city": place["city"],
                "region": place["region"],
                "group": place["group"],
                "temperature": None,
                "humidity": None,
                "time": None
            })

    city_weather_cache = results
    city_weather_cache_time = time.time()

    result = jsonify({"cities": results, "errors": errors})
    result.headers["X-Weather-Cache"] = "MISS"
    return result
@app.route("/api/predictions")
def predictions():
    try:
        if not os.path.exists(CSV_FILE):
            return jsonify({"status": "error", "error": "weather_data.csv was not found."}), 400

        df = pd.read_csv(CSV_FILE)
        required = ["timestamp", "temperature", "humidity", "pressure"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return jsonify({"status": "error", "error": "CSV missing columns.", "missing_columns": missing}), 400

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        for column in ["temperature", "humidity", "pressure"]:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        df = df.dropna(subset=required).sort_values("timestamp").reset_index(drop=True)

        if len(df) < 10:
            return jsonify({"status": "error", "error": "Need at least 10 valid readings."}), 400

        recent = df.tail(10).copy()
        latest = recent.iloc[-1]

        mins_of_day = latest["timestamp"].hour * 60 + latest["timestamp"].minute

        xgb_input = pd.DataFrame([{
            "temperature": float(latest["temperature"]),
            "humidity":    float(latest["humidity"]),
            "pressure":    float(latest["pressure"]),
            "time_sin":    float(np.sin(2 * np.pi * mins_of_day / 1440)),
            "time_cos":    float(np.cos(2 * np.pi * mins_of_day / 1440))
        }])

        lstm_raw    = recent[["temperature", "humidity", "pressure"]].to_numpy(dtype=float)

        result_horizons = {}

        for h in HORIZONS:
            h_result = {}

            # --- XGBoost ---
            if h in xgb_models:
                try:
                    xgb_vals = np.asarray(xgb_models[h].predict(xgb_input)).reshape(-1)
                    h_result["xgboost"] = {
                        "temperature": round(float(xgb_vals[0]), 2),
                        "humidity":    round(float(xgb_vals[1]), 2),
                        "pressure":    round(float(xgb_vals[2]), 2)
                    }
                except Exception as xe:
                    h_result["xgboost"] = {"error": str(xe)}
            else:
                h_result["xgboost"] = {"error": f"XGBoost {h}min model not loaded"}

            # --- LSTM ---
            if h in lstm_models:
                try:
                    lstm_scaled = lstm_x_scalers[h].transform(lstm_raw)
                    lstm_in     = np.expand_dims(lstm_scaled, axis=0)
                    lstm_pred   = lstm_models[h].predict(lstm_in, verbose=0)
                    lstm_vals   = np.asarray(lstm_y_scalers[h].inverse_transform(lstm_pred)).reshape(-1)
                    h_result["lstm"] = {
                        "temperature": round(float(lstm_vals[0]), 2),
                        "humidity":    round(float(lstm_vals[1]), 2),
                        "pressure":    round(float(lstm_vals[2]), 2)
                    }
                except Exception as le:
                    h_result["lstm"] = {"error": str(le)}
            else:
                h_result["lstm"] = {"error": f"LSTM {h}min model not loaded"}

            result_horizons[str(h)] = h_result

        return jsonify({
            "status": "success",
            "latest_reading": {
                "timestamp":   latest["timestamp"].isoformat(),
                "temperature": round(float(latest["temperature"]), 2),
                "humidity":    round(float(latest["humidity"]), 2),
                "pressure":    round(float(latest["pressure"]), 2)
            },
            "horizons": result_horizons,
            # Keep backward-compat keys for old dashboard code
            "xgboost": result_horizons.get("10", {}).get("xgboost", {}),
            "lstm":    result_horizons.get("10", {}).get("lstm", {}),
            "prediction_horizon_minutes": 10,
            "note": "Multi-horizon predictions: 10, 20, 30 minutes ahead."
        })

    except Exception as error:
        app.logger.exception("Weather prediction failed")
        return jsonify({
            "status": "error",
            "error": "Could not generate predictions.",
            "details": str(error)
        }), 500



@app.route("/api/status")
def esp32_status():
    """Returns current ESP32 connection status and last reading info."""
    global latest_local_reading, latest_local_reading_time
    last_time = None
    age_seconds = None
    if latest_local_reading_time:
        import time as _t
        age_seconds = round(_t.time() - latest_local_reading_time, 1)
        from datetime import datetime as _dt
        last_time = _dt.fromtimestamp(latest_local_reading_time).isoformat(timespec="seconds")

    return jsonify({
        "esp32_url": ESP32_URL,
        "serial_port": SERIAL_PORT,
        "serial_available": SERIAL_AVAILABLE,
        "last_reading": latest_local_reading,
        "last_reading_time": last_time,
        "age_seconds": age_seconds,
        "data_logging_active": age_seconds is not None and age_seconds < 90
    })


@app.route("/api/set-esp32-ip", methods=["POST"])
def set_esp32_ip():
    """Update ESP32 IP address at runtime without restarting the server."""
    global ESP32_URL
    data = request.get_json(silent=True)
    if not data or "ip" not in data:
        return jsonify({"error": "Send JSON: {ip: '10.x.x.x'}"}), 400
    new_ip = str(data["ip"]).strip()
    if not new_ip:
        return jsonify({"error": "IP cannot be empty"}), 400
    ESP32_URL = f"http://{new_ip}/api/data"
    print(f"[Config] ESP32_URL updated to: {ESP32_URL}")
    return jsonify({"status": "ok", "esp32_url": ESP32_URL})



if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5001,
        debug=False,
        use_reloader=False
    )
