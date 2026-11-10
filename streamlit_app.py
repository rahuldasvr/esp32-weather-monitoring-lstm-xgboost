
import os
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

# ============================================================
# GROUP 11 - GRAFANA-STYLE STREAMLIT WEATHER DASHBOARD
# ============================================================

st.set_page_config(
    page_title="Weather Dashboard | GROUP 11",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ESP32_URL = os.getenv("ESP32_URL", "http://10.55.192.58/api/data")
CSV_PATH = os.getenv("WEATHER_CSV", "weather_data.csv")
LATITUDE = 10.5276
LONGITUDE = 76.2144
TIMEZONE = "Asia/Kolkata"

BG = "#0b0d0f"
PANEL = "#141619"
GRID = "#34383e"
TEXT = "#d9dee5"
GREEN = "#00f52a"
CYAN = "#00e5ff"
RED = "#ff2638"
BLUE = "#273cff"
YELLOW = "#f5ed00"
MAGENTA = "#d000d9"

st.markdown("""
<style>
.stApp { background:#0b0d0f; color:#e6eaf0; }
[data-testid="stHeader"] { background:#0b0d0f; }
.block-container { padding-top:1rem; max-width:1900px; }
h1,h2,h3 { color:#e6eaf0; }
div[data-testid="stMetric"] {
    background:#141619; border:1px solid #292d33;
    padding:12px; border-radius:5px;
}
div[data-testid="stMetricValue"] { color:#00f52a; }
div[data-testid="stPlotlyChart"] {
    background:#141619; border:1px solid #272b30; border-radius:4px;
}
</style>
""", unsafe_allow_html=True)


def get_num(data, *keys):
    if not isinstance(data, dict):
        return None
    for key in keys:
        value = data.get(key)
        try:
            if value is not None and value != "":
                return float(value)
        except (TypeError, ValueError):
            pass
    return None


def fmt(value, digits=1):
    try:
        if value is None or pd.isna(value):
            return "--"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "--"


def get_live():
    try:
        response = requests.get(ESP32_URL, timeout=3)
        response.raise_for_status()
        data = response.json()
        return {
            "temperature": get_num(data, "temperature", "temp", "Temperature"),
            "humidity": get_num(data, "humidity", "hum", "Humidity"),
            "pressure": get_num(data, "pressure", "pres", "Pressure"),
            "online": True,
            "error": "",
        }
    except Exception as exc:
        return {
            "temperature": None,
            "humidity": None,
            "pressure": None,
            "online": False,
            "error": str(exc),
        }


def get_online_weather():
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "timezone": TIMEZONE,
        "current": (
            "temperature_2m,relative_humidity_2m,dew_point_2m,"
            "apparent_temperature,precipitation,rain,cloud_cover,"
            "pressure_msl,surface_pressure,wind_speed_10m,"
            "wind_direction_10m,weather_code"
        ),
        "hourly": (
            "temperature_2m,relative_humidity_2m,dew_point_2m,"
            "wind_speed_10m,wind_direction_10m,precipitation,rain,"
            "pressure_msl,cloud_cover"
        ),
        "daily": "sunrise,sunset,uv_index_max,precipitation_probability_max,"
                 "temperature_2m_max,temperature_2m_min,weather_code",
        "forecast_days": 7,
    }
    try:
        response = requests.get(url, params=params, timeout=8)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def get_history():
    columns = ["timestamp", "temperature", "humidity", "pressure"]

    if not os.path.exists(CSV_PATH):
        return pd.DataFrame(columns=columns)

    try:
        df = pd.read_csv(CSV_PATH)
        if df.empty:
            return pd.DataFrame(columns=columns)

        aliases = {
            "timestamp": ["timestamp", "time", "datetime", "date"],
            "temperature": ["temperature", "temp", "temperature_c"],
            "humidity": ["humidity", "hum", "relative_humidity"],
            "pressure": ["pressure", "pressure_hpa", "pres"],
        }

        lookup = {str(col).strip().lower(): col for col in df.columns}
        rename = {}

        for target, names in aliases.items():
            for name in names:
                if name in lookup:
                    rename[lookup[name]] = target
                    break

        df = df.rename(columns=rename)

        for col in columns:
            if col not in df.columns:
                df[col] = pd.NA

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        for col in ["temperature", "humidity", "pressure"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return (
            df[columns]
            .dropna(subset=["timestamp"])
            .sort_values("timestamp")
        )
    except Exception as exc:
        st.error(f"Could not read weather history: {exc}")
        return pd.DataFrame(columns=columns)


def chart_layout(fig, height=250, legend=False):
    fig.update_layout(
        height=height,
        margin=dict(l=35, r=12, t=34, b=25),
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=dict(color=TEXT, size=10),
        showlegend=legend,
        legend=dict(orientation="h", y=1.13, x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(
        gridcolor=GRID,
        linecolor=GRID,
        tickfont=dict(color="#aab1bb", size=9),
        showgrid=True,
    )
    fig.update_yaxes(
        gridcolor=GRID,
        linecolor=GRID,
        tickfont=dict(color="#aab1bb", size=9),
        showgrid=True,
    )
    return fig


def make_line(df, column, title, color, unit, height=250):
    fig = go.Figure()

    if not df.empty and column in df.columns:
        fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df[column],
            mode="lines",
            name=title,
            line=dict(color=color, width=1.6),
            fill="tozeroy" if column in ["rain", "precipitation"] else None,
            connectgaps=False,
        ))

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center",
                   font=dict(size=11, color=TEXT)),
        yaxis_title=unit,
    )
    return chart_layout(fig, height)


def make_gauge(value, title, unit, minimum, maximum, color=GREEN):
    numeric_value = value
    if numeric_value is None:
        numeric_value = minimum
        number_color = "#858b94"
        bar_color = "#565c65"
    else:
        number_color = color
        bar_color = color

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=numeric_value,
        title=dict(text=title, font=dict(size=11, color=TEXT)),
        number=dict(
            suffix=unit,
            font=dict(size=20, color=number_color),
        ),
        gauge=dict(
            axis=dict(
                range=[minimum, maximum],
                tickfont=dict(size=8, color="#9aa1aa"),
                tickcolor="#9aa1aa",
            ),
            bar=dict(color=bar_color, thickness=0.25),
            bgcolor="#202328",
            borderwidth=1,
            bordercolor="#30353b",
            steps=[
                dict(
                    range=[minimum, minimum + (maximum - minimum) * 0.65],
                    color="#1b2228",
                ),
                dict(
                    range=[minimum + (maximum - minimum) * 0.65, maximum],
                    color="#202328",
                ),
            ],
            threshold=dict(
                line=dict(color=RED, width=3),
                thickness=0.8,
                value=maximum * 0.92,
            ),
        ),
    ))

    fig.update_layout(
        height=155,
        margin=dict(l=7, r=7, t=25, b=0),
        paper_bgcolor=PANEL,
        font=dict(color=TEXT),
    )
    return fig


def make_wind_speed(df):
    fig = go.Figure()

    if "wind_speed_10m" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["wind_speed_10m"],
            name="Wind speed", mode="lines",
            line=dict(color=GREEN, width=1.5),
        ))

    fig.update_layout(
        title=dict(text="Wind Speed", x=0.5, xanchor="center",
                   font=dict(size=11)),
        yaxis_title="km/h",
    )
    return chart_layout(fig, 250)


def make_wind_direction(df):
    fig = go.Figure()

    if "wind_direction_10m" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df["wind_direction_10m"],
            name="Wind direction",
            mode="markers",
            marker=dict(color=YELLOW, size=4),
        ))

    fig.update_layout(
        title=dict(text="Wind Direction", x=0.5, xanchor="center",
                   font=dict(size=11)),
        yaxis=dict(
            title="Degrees",
            range=[0, 360],
            tickvals=[0, 90, 180, 270, 360],
        ),
    )
    return chart_layout(fig, 250)


def make_rain(df):
    fig = go.Figure()

    for column, color in [
        ("rain", MAGENTA),
        ("precipitation", CYAN),
    ]:
        if column in df.columns and df[column].notna().any():
            fig.add_trace(go.Scatter(
                x=df["timestamp"],
                y=df[column],
                name=column.title(),
                mode="lines",
                line=dict(color=color, width=1.5),
                fill="tozeroy",
            ))

    fig.update_layout(
        title=dict(text="Rain", x=0.5, xanchor="center",
                   font=dict(size=11)),
        yaxis_title="mm",
    )
    return chart_layout(fig, 250, True)


# ------------------------------------------------------------
# HEADER / CONTROLS
# ------------------------------------------------------------

header_left, header_right = st.columns([3, 2])

with header_left:
    st.markdown("## ▦  Weather")
    st.caption("GROUP 11 · IoT Smart Weather Monitoring Dashboard")

with header_right:
    selected_range = st.selectbox(
        "History range",
        ["Last 1 hour", "Last 6 hours", "Last 12 hours",
         "Last 24 hours", "All records"],
        index=2,
    )
    st.caption("Use the refresh button in the Streamlit toolbar to reload data.")

live = get_live()
online = get_online_weather()
history = get_history()

if not history.empty and selected_range != "All records":
    hours = {
        "Last 1 hour": 1,
        "Last 6 hours": 6,
        "Last 12 hours": 12,
        "Last 24 hours": 24,
    }[selected_range]

    latest = history["timestamp"].max()
    history = history[
        history["timestamp"] >= latest - pd.Timedelta(hours=hours)
    ]

current = (online or {}).get("current", {})

temperature = live["temperature"]
humidity = live["humidity"]
pressure = live["pressure"]

if temperature is None:
    temperature = get_num(current, "temperature_2m")
if humidity is None:
    humidity = get_num(current, "relative_humidity_2m")
if pressure is None:
    pressure = get_num(current, "surface_pressure", "pressure_msl")

dew_point = get_num(current, "dew_point_2m")

status1, status2, status3 = st.columns([1, 1, 2])

with status1:
    st.markdown("🟢 **ESP32 Connected**" if live["online"]
                else "🔴 **ESP32 Offline**")

with status2:
    st.markdown("🌐 **Online Weather Available**" if online
                else "⚠️ **Online Weather Unavailable**")

with status3:
    st.caption(
        f"Updated {pd.Timestamp.now().strftime('%d %b %Y, %H:%M:%S')} "
        f"· {len(history)} history records"
    )

if not live["online"]:
    st.warning(
        "Could not reach the ESP32. CSV history and online weather can still be displayed. "
        f"Connection details: {live['error']}"
    )

# ------------------------------------------------------------
# GRAFANA-LIKE DASHBOARD GRID
# ------------------------------------------------------------

left, center, right = st.columns([0.78, 2.45, 2.45], gap="small")

with left:
    st.plotly_chart(
        make_gauge(temperature, "Temperature", "°C", -10, 50),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    st.plotly_chart(
        make_gauge(humidity, "Humidity", "%", 0, 100),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    st.plotly_chart(
        make_gauge(dew_point, "Dew Point", "°C", -10, 40, CYAN),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    st.plotly_chart(
        make_gauge(pressure, "Pressure", " hPa", 950, 1050),
        use_container_width=True,
        config={"displayModeBar": False},
    )

with center:
    st.plotly_chart(
        make_line(history, "temperature", "Temperature", RED, "°C"),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    st.plotly_chart(
        make_line(history, "humidity", "Humidity", CYAN, "%"),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    st.plotly_chart(
        make_line(history, "pressure", "Barometer", GREEN, "hPa"),
        use_container_width=True,
        config={"displayModeBar": False},
    )

with right:
    hourly = (online or {}).get("hourly", {})

    if hourly and hourly.get("time"):
        hourly_df = pd.DataFrame(hourly)
        hourly_df["timestamp"] = pd.to_datetime(
            hourly_df["time"], errors="coerce"
        )
        hourly_df = hourly_df.dropna(subset=["timestamp"]).head(48)

        st.plotly_chart(
            make_wind_speed(hourly_df),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.plotly_chart(
            make_wind_direction(hourly_df),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        if "pressure_msl" in hourly_df.columns:
            st.plotly_chart(
                make_line(
                    hourly_df, "pressure_msl",
                    "Online Barometer", GREEN, "hPa"
                ),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        else:
            st.plotly_chart(
                make_rain(hourly_df),
                use_container_width=True,
                config={"displayModeBar": False},
            )
    else:
        st.plotly_chart(
            make_wind_speed(pd.DataFrame(columns=["timestamp", "wind_speed_10m"])),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.plotly_chart(
            make_wind_direction(pd.DataFrame(columns=["timestamp", "wind_direction_10m"])),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.plotly_chart(
            make_rain(pd.DataFrame()),
            use_container_width=True,
            config={"displayModeBar": False},
        )

# ------------------------------------------------------------
# 7-DAY FORECAST
# ------------------------------------------------------------

st.markdown("### 7-Day Forecast")

daily = (online or {}).get("daily", {})

if daily and daily.get("time"):
    forecast = pd.DataFrame({
        "Date": daily.get("time", []),
        "Maximum °C": daily.get("temperature_2m_max", []),
        "Minimum °C": daily.get("temperature_2m_min", []),
        "UV Index": daily.get("uv_index_max", []),
        "Rain Chance %": daily.get("precipitation_probability_max", []),
        "Sunrise": daily.get("sunrise", []),
        "Sunset": daily.get("sunset", []),
    })
    st.dataframe(forecast, use_container_width=True, hide_index=True)
else:
    st.caption("Online daily forecast is currently unavailable.")

# ------------------------------------------------------------
# HISTORY TABLE
# ------------------------------------------------------------

with st.expander("View recent sensor records"):
    if history.empty:
        st.info("No CSV history records are available for this range.")
    else:
        st.dataframe(
            history.sort_values("timestamp", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

st.caption(
    "GROUP 11 · Temperature and humidity may fall back to Open-Meteo if ESP32 is offline. "
    "Historical graphs use weather_data.csv. Missing pressure values are left blank."
)
