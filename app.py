# ==========================================================
# APP.PY - ABUJA ONLY
# Live Dashboard for Abuja
# ==========================================================
import random
import os
import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# --- REPRODUCIBILITY ---
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
os.environ['PYTHONHASHSEED'] = str(RANDOM_STATE)

st.set_page_config(page_title="Abuja Solar Forecast", page_icon="☀️", layout="wide")
st.title("☀️ Abuja Solar Power Forecast")
st.markdown("Live solar prediction using NASA POWER API + Machine Learning | Capstone Project")

LAT, LON = 9.08, 7.49  # Abuja

# --- FETCH LIVE DATA ---
@st.cache_data(ttl=3600)
def fetch_live_nasa_data(days_back=35):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": "ALLSKY_SFC_SW_DWN,T2M,RH2M,WS2M,PS",
        "community": "RE",
        "longitude": LON, "latitude": LAT,
        "start": start_date.strftime("%Y%m%d"),
        "end": end_date.strftime("%Y%m%d"),
        "format": "JSON"
    }
    try:
        response = requests.get(url, params=params, timeout=30)
        if response.status_code != 200:
            st.error(f"NASA API Error: {response.status_code}")
            return None
        data = response.json()
        df = pd.DataFrame(data['properties']['parameter'])
        df.index = pd.to_datetime(df.index, format='%Y%m%d')
        df = df.reset_index().rename(columns={'index': 'date'})
        df = df.rename(columns={
            'ALLSKY_SFC_SW_DWN': 'irradiance', 'T2M': 'temperature',
            'RH2M': 'humidity', 'WS2M': 'wind_speed', 'PS': 'pressure'
        })
        df = df.replace(-999, np.nan)
        df = df.dropna()
        df['month'] = df['date'].dt.month
        df['dayofyear'] = df['date'].dt.dayofyear
        for lag in [1, 2, 3, 7]:
            df[f'irradiance_lag_{lag}'] = df['irradiance'].shift(lag)
        df['irradiance_roll_7'] = df['irradiance'].rolling(7).mean()
        df = df.dropna()
        return df
    except Exception as e:
        st.error(f"Failed to fetch live data: {e}")
        return None

# --- NEW - LIVE WEATHER API (Open-Meteo, free, no key) ---
@st.cache_data(ttl=600)
def fetch_live_weather_openmeteo():
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "current": "temperature_2m,relative_humidity_2m,cloud_cover,shortwave_radiation,wind_speed_10m",
        "hourly": "shortwave_radiation,temperature_2m,cloud_cover",
        "timezone": "Africa/Lagos",
        "forecast_days": 1
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code != 200:
            return None
        j = r.json()
        cur = j.get("current", {})
        return {
            "time": cur.get("time", ""),
            "temp_live": cur.get("temperature_2m"),
            "humidity_live": cur.get("relative_humidity_2m"),
            "cloud_live": cur.get("cloud_cover"),
            "radiation_live": cur.get("shortwave_radiation"),
            "wind_live": cur.get("wind_speed_10m"),
        }
    except Exception:
        return None

@st.cache_resource
def load_trained_model():
    try:
        model = joblib.load('best_model.pkl')
        scaler_x = joblib.load('scaler_x.pkl')
        features = joblib.load('feature_list.pkl')
        return model, scaler_x, features, True
    except Exception:
        return None, None, None, False

# --- SIDEBAR ---
st.sidebar.header("🎛️ User Controls")
location = st.sidebar.selectbox("📍 Location", ["Abuja, FCT"])
forecast_days = st.sidebar.slider("Forecast Days", 1, 7, 3)
user_type = st.sidebar.selectbox("👤 I am a...", ["Grid Operator", "Mini-Grid Operator", "Homeowner"])
get_forecast = st.sidebar.button("🚀 Get Live Forecast")

# --- NEW - BATTERY INPUTS IN SIDEBAR ---
st.sidebar.markdown("---")
st.sidebar.header("🔋 Battery Advice")
battery_size = st.sidebar.number_input("How much power can your battery hold? (kWh)", min_value=1.0, max_value=50.0, value=5.0, step=0.5, help="When full, how much power is inside it? Example: Small house = 5, Big house = 10. Check the label on your battery.")
battery_age_months = st.sidebar.number_input("When did you buy your battery? (months ago)", min_value=0, max_value=120, value=12, step=1, help="Example: Bought last year = 12, Bought 2 years ago = 24. Just guess if you don't remember.")
daily_load = st.sidebar.number_input("How much light will you use today? (kWh)", min_value=1.0, max_value=50.0, value=5.0, step=0.5, help="Example: Fridge + lights + fan for one day = about 5")

# --- NEW - BATTERY HEALTH - LAYMAN VERSION (no word "cycles" shown) ---
st.sidebar.markdown("---")
st.sidebar.header("🩺 Battery Health")
battery_age_months = st.sidebar.number_input("How long have you been using this battery? (months ago)", min_value=0, max_value=120, value=12, step=1, help="Example: Using it for 1 year = 12, Using it for 2 years = 24")
charge_how_often = st.sidebar.selectbox(
    "How often do you charge it?",
    ["Every day", "Every 2 days", "Every 3-4 days", "Only when NEPA goes off (about once a week)"],
    help="Just pick what is closest to your normal life. We will estimate the rest."
)
# Convert layman answer to cycles behind the scenes - user never sees "cycles"
# 1 month = ~30 days
days_old = battery_age_months * 30
if charge_how_often == "Every day":
    battery_cycles = days_old / 1
elif charge_how_often == "Every 2 days":
    battery_cycles = days_old / 2
elif charge_how_often == "Every 3-4 days":
    battery_cycles = days_old / 3.5
else:  # Only when NEPA goes off
    battery_cycles = days_old / 7

if not get_forecast:
    st.info("👈 Click **Get Live Forecast** to fetch live NASA data for Abuja")
    st.markdown("### ☀️ What This System Does")
    st.markdown("""
    - Fetches **live weather** for Abuja from NASA POWER API
    - Uses **Machine Learning** (Linear Regression vs XGBoost vs Random Forest) to predict solar power
    - Shows **hourly forecast**, **30-day trend**, and **recommendation** for your user type
    - Random State = 42 fixed, so results are reproducible
    """)
    st.stop()

# --- PREDICTION ---
with st.spinner("Fetching live NASA data for Abuja..."):
    df = fetch_live_nasa_data()

if df is None or len(df) == 0:
    st.error("No live data. Check internet and try again.")
    st.stop()

model, scaler_x, features, model_loaded = load_trained_model()
if model_loaded:
    x_live = df[features].values
    x_scaled = scaler_x.transform(x_live)
    pred_pv = model.predict(x_scaled)[-1]
    pred_pv = max(0, pred_pv)
    st.success("✅ Model loaded: Prediction using trained ML model")
else:
    last = df.iloc[-1]
    pred_pv = last['irradiance'] * 0.85 * (1 - 0.004 * (last['temperature'] - 25))
    pred_pv = max(0, pred_pv)
    st.warning("⚠️ best_model.pkl not found. Using physics formula as demo.")

# --- NEW - FETCH LIVE OPEN-METEO ---
live = fetch_live_weather_openmeteo()
if live is not None:
    st.subheader("🌤️ Live Right Now in Abuja - Open-Meteo API")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("☀️ Live Radiation", f"{live['radiation_live']:.0f} W/m²" if live['radiation_live'] is not None else "—")
    c2.metric("🌡️ Live Temperature", f"{live['temp_live']:.1f} °C" if live['temp_live'] is not None else "—")
    c3.metric("☁️ Cloud Cover", f"{live['cloud_live']:.0f} %" if live['cloud_live'] is not None else "—")
    c4.metric("💨 Wind", f"{live['wind_live']:.1f} km/h" if live['wind_live'] is not None else "—")
    if live['radiation_live'] is not None and live['radiation_live'] < 200 and pred_pv > 2.5:
        st.warning("⚠️ Live sky is cloudy right now, but daily forecast is high. Expect lower power for the next 1-2 hours, then it should recover.")
    st.caption(f"Live update: {live['time']} | Source: Open-Meteo (free, no key) | Location: Abuja 9.08, 7.49")
else:
    st.info("ℹ️ Live Open-Meteo data unavailable right now. Showing NASA forecast only.")

# --- 24H CURVE ---
hours = np.arange(24)
solar_curve = np.maximum(0, np.sin((hours - 6) / 12 * np.pi)) ** 1.5
hourly_forecast = np.round(solar_curve * pred_pv * 1.8, 2)
avg_power = hourly_forecast.mean()

# --- METRICS ---
st.subheader(f"📊 Live Forecast for Abuja - {datetime.now().strftime('%d %b %Y')}")
col1, col2, col3, col4 = st.columns(4)
last = df.iloc[-1]
col1.metric("☀️ Irradiance Today", f"{last['irradiance']:.2f} kWh/m²/day")
col2.metric("🌡️ Temperature", f"{last['temperature']:.1f} °C")
col3.metric("💧 Humidity", f"{last['humidity']:.1f} %")
col4.metric("⚡ Predicted PV Power", f"{pred_pv:.2f} kW")

# --- GRAPH 1 ---
st.subheader("📈 24-Hour Solar Forecast - Abuja")
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(hours, hourly_forecast, marker='o', color='#FFA500', linewidth=2, label='Predicted PV Power')
ax.fill_between(hours, hourly_forecast, alpha=0.3, color='#FFA500')
ax.set_xlabel("Hour of Day")
ax.set_ylabel("PV Power (kW)")
ax.set_title("Hourly Solar Power Forecast - Abuja")
ax.set_xticks(range(0, 24))
ax.set_xticklabels([f"{h}:00" for h in range(24)], rotation=45)
ax.grid(alpha=0.3)
ax.legend()
st.pyplot(fig)

# --- GRAPH 2 ---
st.subheader("📉 Last 30 Days Irradiance Trend - Abuja")
fig2, ax2 = plt.subplots(figsize=(12, 3))
ax2.plot(df['date'].tail(30), df['irradiance'].tail(30), color='#1E90FF', marker='o')
ax2.set_ylabel("Irradiance")
ax2.set_title("Last 30 Days Sunlight - Abuja")
ax2.grid(alpha=0.3)
plt.xticks(rotation=45)
st.pyplot(fig2)

# --- TABLE ---
st.subheader("📋 Hourly Breakdown")
hourly_df = pd.DataFrame({
    'Hour': [f"{h}:00" for h in hours],
    'Predicted PV Power (kW)': hourly_forecast,
    'Status': ['Night' if v == 0 else 'Low' if v < 1 else 'Moderate' if v < 2.5 else 'High' for v in hourly_forecast]
})
st.dataframe(hourly_df, use_container_width=True)

# --- RECOMMENDATION ---
st.subheader("💡 Recommendation")
if user_type == "Grid Operator":
    if avg_power > 2.5:
        st.success("✅ High solar in Abuja. You can reduce reliance on grid / diesel today.")
    elif avg_power > 1.2:
        st.warning("⚠️ Moderate solar in Abuja. Keep backup ready, use solar + grid mix.")
    else:
        st.error("🔴 Low solar in Abuja. Expect cloudy / rainy. Switch to grid / battery backup.")
elif user_type == "Mini-Grid Operator":
    if avg_power > 2.5:
        st.success("✅ High solar in Abuja. Charge batteries fully, you can serve more households today.")
    elif avg_power > 1.2:
        st.warning("⚠️ Moderate solar in Abuja. Manage load, prioritize essential loads in evening.")
    else:
        st.error("🔴 Low solar in Abuja. Conserve battery, use diesel backup if available.")
else:
    if avg_power > 2.5:
        st.success("✅ Great day in Abuja! Run washing machine, pump water, charge devices with solar.")
    elif avg_power > 1.2:
        st.warning("⚠️ Okay day in Abuja. Use solar in afternoon, save battery for night.")
    else:
        st.error("🔴 Cloudy day in Abuja. Don't rely on solar alone, keep PHCN / generator ready.")

# --- NEW - BATTERY ADVICE ---
st.subheader("🔋 Battery Advice - What To Do Now")
now_hour = datetime.now().hour
remaining_energy = float(hourly_forecast[now_hour:].sum())
battery_now_kwh = battery_size * (battery_now_pct / 100.0)
battery_needed_kwh = battery_size - battery_now_kwh
solar_after_load = max(0, remaining_energy - daily_load)
hours_to_full = (battery_needed_kwh / max(0.1, pred_pv)) if pred_pv > 0 else 99

b1, b2, b3 = st.columns(3)
b1.metric("🔋 Battery Now", f"{battery_now_kwh:.1f} / {battery_size:.1f} kWh ({battery_now_pct}%)")
b2.metric("☀️ Solar Left Today", f"{remaining_energy:.1f} kWh")
b3.metric("⏱️ Time to Full (at current power)", f"{hours_to_full:.1f} hrs" if hours_to_full < 20 else "Not today")

if pred_pv > 3.5:
    st.success(f"✅ CHARGE NOW - Best window 11am to 3pm. You have ~{solar_after_load:.1f} kWh left after your load. Battery can be full in ~{hours_to_full:.1f} hrs. Run freezer / washing machine / pump now.")
elif pred_pv > 1.5:
    st.warning(f"⚠️ CHARGE SLOWLY - Moderate sun. You have ~{solar_after_load:.1f} kWh left. Charge battery slowly, run only essential loads. Save battery for evening.")
else:
    st.error(f"🔴 SAVE BATTERY - Low sun. Only ~{solar_after_load:.1f} kWh left. Don't charge heavy loads. Keep battery for lights / fan / phone at night. Use grid / generator if you must.")

if solar_after_load >= battery_needed_kwh:
    st.info(f"💡 Good news: Today's remaining sun ({solar_after_load:.1f} kWh) is enough to fill your battery ({battery_needed_kwh:.1f} kWh needed).")
else:
    st.info(f"💡 Heads up: Today's remaining sun ({solar_after_load:.1f} kWh) is NOT enough to fill your battery ({battery_needed_kwh:.1f} kWh needed). Prioritize essentials.")

# --- NEW - BATTERY HEALTH - LAYMAN VERSION ---
st.subheader("🩺 Battery Health - How Strong Is Your Battery?")
# Same formula, but cycles now come from layman dropdown, not typed number
# Health = 100% - (Age x 0.25%) - (Cycles x 0.02%)
health_loss = (battery_age_months * 0.25) + (battery_cycles * 0.02)
health_pct = max(50.0, 100.0 - health_loss)
real_capacity = battery_size * (health_pct / 100.0)
lost_capacity = battery_size - real_capacity

h1, h2, h3 = st.columns(3)
h1.metric("🩺 Battery Health", f"{health_pct:.0f}%")
h2.metric("🔋 Real Capacity Now", f"{real_capacity:.2f} kWh (was {battery_size:.1f} kWh)")
h3.metric("📉 Capacity Lost", f"{lost_capacity:.2f} kWh")

st.progress(int(health_pct))

if health_pct >= 85:
    st.success(f"✅ GOOD HEALTH ({health_pct:.0f}%) - Your battery is still strong. It holds {real_capacity:.2f} kWh. Keep using normally. Avoid draining to 0% to keep it healthy.")
elif health_pct >= 70:
    st.warning(f"⚠️ FAIR HEALTH ({health_pct:.0f}%) - Your battery has lost {lost_capacity:.2f} kWh. Tip: Charge to 80% only, don't drain below 20%. It will still work, but won't last all night like before.")
else:
    st.error(f"🔴 WEAK BATTERY ({health_pct:.0f}%) - Your battery now holds only {real_capacity:.2f} kWh out of {battery_size:.1f} kWh. Consider replacing soon. Don't rely on it alone for the full night - keep grid / generator backup.")

st.caption(f"Based on: {battery_age_months} months old, charging {charge_how_often.lower()}. Estimate only, not a lab test.")

# --- DOWNLOAD ---
csv = hourly_df.to_csv(index=False)
st.download_button("⬇️ Download Forecast for Abuja (CSV)", csv, "solar_forecast_abuja.csv", "text/csv")
st.caption("Data: NASA POWER API (Live) + Open-Meteo Live Weather | Models: Linear Regression vs XGBoost vs Random Forest | Location: Abuja, FCT (9.08, 7.49) | Seed: 42 for reproducibility")