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
from zoneinfo import ZoneInfo

# --- LAGOS TIME ---
LAGOS_TZ = ZoneInfo("Africa/Lagos")
def lagos_now():
    return datetime.now(LAGOS_TZ)
def lagos_hour():
    return datetime.now(LAGOS_TZ).hour

# --- REPRODUCIBILITY ---
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
os.environ['PYTHONHASHSEED'] = str(RANDOM_STATE)

st.set_page_config(page_title="Abuja Solar Forecast", page_icon="☀️", layout="wide")

# ==========================================================
# FIX: SIDEBAR NOTICEABLE ON PHONE, CLEAN ON LAPTOP
# ==========================================================
st.markdown("""
<style>
.phone-banner { display: none; }
@media (max-width: 768px) {
    [data-testid="collapsedControl"] {
        font-size: 28px !important;
        background-color: #FFA500 !important;
        border-radius: 12px !important;
        padding: 8px 12px !important;
        box-shadow: 0 0 15px #FFA500 !important;
        animation: pulse 1.5s infinite !important;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.15); }
        100% { transform: scale(1); }
    }
    .phone-banner { display: block; }
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="phone-banner">', unsafe_allow_html=True)
st.warning("👈 TAP **>>** TOP-LEFT TO OPEN CONTROLS - Change Battery, User Type, Forecast Days")
st.markdown('</div>', unsafe_allow_html=True)

@st.dialog("👈 Open Controls First!")
def show_controls_popup():
    st.markdown("""
    ### To use this app on phone:
    1. Tap **>>** at the TOP-LEFT corner
    2. Sidebar will open
    3. Change:
       - 🔋 Battery - Pick your battery type, how many, inverter lights
       - 👤 I am a... (Homeowner / Grid Operator)
       - 📅 Forecast Days
    4. Tap **🚀 Get Live Forecast**
    
    On laptop, sidebar is already open on the left.
    """)
    if st.button("✅ Got it, Show Me Forecast", use_container_width=True):
        st.rerun()

if "popup_shown" not in st.session_state:
    st.session_state.popup_shown = True
    show_controls_popup()
# ==========================================================

st.title("☀️ Abuja Solar Power Forecast")
st.markdown("Live solar prediction using NASA POWER API + Machine Learning | Capstone Project")

LAT, LON = 9.08, 7.49  # Abuja

# --- FETCH LIVE DATA ---
@st.cache_data(ttl=3600)
def fetch_live_nasa_data(days_back=35):
    end_date = lagos_now()
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

# --- LIVE WEATHER API (Open-Meteo) ---
@st.cache_data(ttl=600)
def fetch_live_weather_openmeteo():
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "current": "temperature_2m,relative_humidity_2m,cloud_cover,shortwave_radiation,wind_speed_10m",
        "hourly": "shortwave_radiation,temperature_2m,cloud_cover",
        "timezone": "Africa/Lagos",
        "forecast_days": 2
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code != 200:
            return None
        j = r.json()
        cur = j.get("current", {})
        hourly = j.get("hourly", {})
        return {
            "time": cur.get("time", ""),
            "temp_live": cur.get("temperature_2m"),
            "humidity_live": cur.get("relative_humidity_2m"),
            "cloud_live": cur.get("cloud_cover"),
            "radiation_live": cur.get("shortwave_radiation"),
            "wind_live": cur.get("wind_speed_10m"),
            "hourly_time": hourly.get("time", []),
            "hourly_radiation": hourly.get("shortwave_radiation", []),
            "hourly_temp": hourly.get("temperature_2m", []),
            "hourly_cloud": hourly.get("cloud_cover", []),
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

# ==========================================================
# CHANGED: BATTERY ADVICE - LAYMAN VERSION, AUTO-CALCULATED
# No more kWh typing. Pick what you know, we calculate.
# ==========================================================
st.sidebar.markdown("---")
st.sidebar.header("🔋 Battery Advice")
st.sidebar.caption("Don't worry about kWh - just pick what you have, we calculate it for you.")

battery_type = st.sidebar.selectbox(
    "What battery do you have? (Check label on battery)",
    [
        "200Ah 12V - Big Tubular / Inverter Battery (Common)",
        "220Ah 12V - Big Tubular",
        "150Ah 12V - Medium Battery",
        "100Ah 12V - Small Battery",
        "I don't know - Use Small House Default"
    ]
)

if "200Ah" in battery_type:
    kwh_per_battery = 2.4
elif "220Ah" in battery_type:
    kwh_per_battery = 2.64
elif "150Ah" in battery_type:
    kwh_per_battery = 1.8
elif "100Ah" in battery_type:
    kwh_per_battery = 1.2
else:
    kwh_per_battery = 2.4

num_batteries = st.sidebar.selectbox("How many batteries do you have?", [1, 2, 3, 4], index=1)
battery_size = kwh_per_battery * num_batteries
st.sidebar.success(f"✅ Your battery can hold: **{battery_size:.1f} kWh** - Calculated automatically")

battery_bars = st.sidebar.selectbox(
    "How full is your battery now? (Look at inverter lights)",
    ["🟩🟩🟩🟩 Full - 4 lights", "🟩🟩🟩⬜ Almost Full - 3 lights", "🟩🟩⬜⬜ Half - 2 lights", "🟩⬜⬜⬜ Low - 1 light", "⬜⬜⬜⬜ Empty - No light / Red light"],
    index=2
)
if "4 lights" in battery_bars:
    battery_now_pct = 100
elif "3 lights" in battery_bars:
    battery_now_pct = 75
elif "2 lights" in battery_bars:
    battery_now_pct = 50
elif "1 light" in battery_bars:
    battery_now_pct = 25
else:
    battery_now_pct = 5
st.sidebar.caption(f"That's about {battery_now_pct}% full")

st.sidebar.markdown("What will you use today? (Tick all)")
use_fridge = st.sidebar.checkbox("🧊 Fridge", value=True)
use_fans = st.sidebar.checkbox("🌀 Fans (2 fans)", value=True)
use_lights = st.sidebar.checkbox("💡 Lights (5 bulbs)", value=True)
use_tv = st.sidebar.checkbox("📺 TV")
use_pump = st.sidebar.checkbox("💧 Water Pump")
use_washing = st.sidebar.checkbox("👕 Washing Machine")

daily_load = 0.0
if use_fridge: daily_load += 1.5
if use_fans: daily_load += 1.0
if use_lights: daily_load += 0.8
if use_tv: daily_load += 0.8
if use_pump: daily_load += 1.0
if use_washing: daily_load += 1.0
if daily_load == 0: daily_load = 2.0
st.sidebar.success(f"✅ You will use about: **{daily_load:.1f} kWh today** - Calculated automatically")
# ==========================================================
# END CHANGED - BATTERY ADVICE
# ==========================================================

# ==========================================================
# UNCHANGED: BATTERY HEALTH - LEFT EXACTLY AS YOU HAD IT
# ==========================================================
st.sidebar.markdown("---")
st.sidebar.header("🩺 Battery Health")
battery_age_months = st.sidebar.number_input("How long have you been using this battery? (months ago)", min_value=0, max_value=120, value=12, step=1, help="Example: Using it for 1 year = 12, Using it for 2 years = 24")
charge_how_often = st.sidebar.selectbox(
    "How often do you charge it?",
    ["Every day", "Every 2 days", "Every 3-4 days", "Only when NEPA goes off (about once a week)"],
    help="Just pick what is closest to your normal life. We will estimate the rest."
)
days_old = battery_age_months * 30
if charge_how_often == "Every day":
    battery_cycles = days_old / 1
elif charge_how_often == "Every 2 days":
    battery_cycles = days_old / 2
elif charge_how_often == "Every 3-4 days":
    battery_cycles = days_old / 3.5
else:
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

# --- FETCH LIVE OPEN-METEO ---
live = fetch_live_weather_openmeteo()
now_hour_live = lagos_hour()
is_evening = now_hour_live >= 17 or now_hour_live < 6

if live is not None:
    st.subheader("🌤️ Live Right Now in Abuja - Open-Meteo API")
    c1, c2, c3, c4 = st.columns(4)
    live_radiation = live['radiation_live'] if live['radiation_live'] is not None else 0
    with c1:
        st.metric("☀️ Live Radiation", f"{live_radiation:.0f} W/m²")
        if is_evening:
            st.caption("🌙 Sun is setting - No more charging today")
        elif live_radiation >= 600:
            st.caption("🔥 Very Hot Sun - Charging very fast")
        elif live_radiation >= 300:
            st.caption("☀️ Hot Sun - Charging well")
        elif live_radiation >= 100:
            st.caption("🌤️ Small Sun - Charging slowly")
        else:
            st.caption("🌑 No Sun - Not charging")
    live_temp = live['temp_live'] if live['temp_live'] is not None else 0
    with c2:
        st.metric("🌡️ Live Temp (Right Now)", f"{live_temp:.1f} °C")
        if live_temp >= 35:
            st.caption("🥵 Very Hot")
        elif live_temp >= 28:
            st.caption("☀️ Hot")
        elif live_temp >= 22:
            st.caption("😊 Fine / Cool")
        else:
            st.caption("❄️ Cold")
    live_cloud = live['cloud_live'] if live['cloud_live'] is not None else 0
    with c3:
        st.metric("☁️ Cloud Cover", f"{live_cloud:.0f} %")
        if is_evening:
            st.caption("🌙 Night time - Cloud does not matter now")
        elif live_cloud <= 10:
            st.caption("☀️ Clear Sky - No cloud")
        elif live_cloud <= 50:
            st.caption("⛅ Small Cloud")
        else:
            st.caption("☁️ Plenty Cloud - Sun is blocked")
    live_wind = live['wind_live'] if live['wind_live'] is not None else 0
    with c4:
        st.metric("💨 Wind", f"{live_wind:.1f} km/h")
        if live_wind >= 30:
            st.caption("💨 Strong Wind")
        elif live_wind >= 10:
            st.caption("🍃 Small Wind")
        else:
            st.caption("😊 Calm - No wind")
    if is_evening:
        st.info("🌙 Evening / Night time - Sun is setting / already down. No more charging today. Use your battery / grid for tonight. Tomorrow sun will come again.")
    elif live['radiation_live'] is not None and live['radiation_live'] < 200 and pred_pv > 2.5:
        st.warning("⚠️ Live sky is cloudy right now, but daily forecast is high. Expect lower power for the next 1-2 hours, then it should recover.")
    st.caption(f"Live update: {live['time']} | Source: Open-Meteo (free, no key) | Location: Abuja 9.08, 7.49")
else:
    st.info("ℹ️ Live Open-Meteo data unavailable right now. Showing NASA forecast only. System still predicts from NASA + ML model (fallback).")

# --- 24H CURVE ---
hours = np.arange(24)
solar_curve = np.maximum(0, np.sin((hours - 6) / 12 * np.pi)) ** 1.5
hourly_forecast = np.round(solar_curve * pred_pv, 2)

if live is not None and live.get("hourly_radiation"):
    try:
        real_hourly = []
        for i in range(24):
            if i < len(live["hourly_radiation"]):
                rad = live["hourly_radiation"][i] or 0
                temp_h = live["hourly_temp"][i] if i < len(live.get("hourly_temp", [])) else 25
                temp_factor = 1 - 0.004 * (temp_h - 25)
                pv_h = (rad / 1000.0) * pred_pv * max(0.5, temp_factor)
                real_hourly.append(round(max(0, pv_h), 2))
            else:
                real_hourly.append(0.0)
        if sum(real_hourly) > 0:
            hourly_forecast = np.array(real_hourly)
    except Exception:
        pass

avg_power = hourly_forecast.mean()

# --- PV NOW ---
pv_now = float(hourly_forecast[lagos_hour()]) if len(hourly_forecast) == 24 else float(pred_pv)
if live is not None and live.get("radiation_live") is not None:
    live_rad = live["radiation_live"] or 0
    live_temp_for_pv = live.get("temp_live") or 25
    temp_factor_live = 1 - 0.004 * (live_temp_for_pv - 25)
    pv_from_live = (live_rad / 1000.0) * pred_pv * max(0.5, temp_factor_live)
    pv_now = round(max(0, pv_from_live), 2)

# --- METRICS ---
st.subheader(f"📊 Live Forecast for Abuja - {lagos_now().strftime('%d %b %Y')}")
col1, col2, col3, col4 = st.columns(4)
last = df.iloc[-1]
irradiance_today = last['irradiance']
pv_output = pv_now
now_hour_forecast = lagos_hour()
is_evening_forecast = now_hour_forecast >= 17 or now_hour_forecast < 6

with col1:
    st.metric("☀️ Sunlight Today", f"{irradiance_today:.2f} kWh")
    if is_evening_forecast:
        if irradiance_today >= 5.5:
            st.caption("☀️ Today Was Sunny - Sun has set")
        elif irradiance_today >= 3.5:
            st.caption("⛅ Today Was Fair - Sun has set")
        else:
            st.caption("🌧️ Today Was Cloudy - Sun has set")
    else:
        if irradiance_today >= 5.5:
            st.caption("☀️ Sunny Day - Plenty sun")
        elif irradiance_today >= 3.5:
            st.caption("⛅ Fair Day - Some sun")
        else:
            st.caption("🌧️ Cloudy Day - Less sun")
with col2:
    st.metric("🌡️ Temp (Today Avg)", f"{last['temperature']:.1f} °C")
    if last['temperature'] >= 35:
        st.caption("🥵 Very Hot")
    elif last['temperature'] >= 28:
        st.caption("☀️ Hot")
    elif last['temperature'] >= 22:
        st.caption("😊 Fine / Cool")
    else:
        st.caption("❄️ Cold")
with col3:
    st.metric("💧 Humidity (Today Avg)", f"{last['humidity']:.1f} %")
    if last['humidity'] >= 70:
        st.caption("💧 Very Damp")
    elif last['humidity'] >= 40:
        st.caption("😊 Normal")
    else:
        st.caption("🏜️ Dry Air")
with col4:
    st.metric("⚡ PV Power Now", f"{pv_output:.2f} kW")
    if is_evening_forecast:
        st.caption("🌙 Sun has set - No power now")
    elif pv_output >= 2.5:
        st.caption("⭐ Excellence - Very well")
    elif pv_output >= 1.5:
        st.caption("👍 Good - Working okay")
    elif pv_output > 0:
        st.caption("⚠️ Low - Not much")
    else:
        st.caption("🌑 No power - Night")

# --- GRAPH 1 ---
st.subheader("📈 24-Hour Solar Forecast - Abuja (From Open-Meteo Hourly)")
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(hours, hourly_forecast, marker='o', color='#FFA500', linewidth=2, label='Predicted PV Power (Open-Meteo)')
ax.fill_between(hours, hourly_forecast, alpha=0.3, color='#FFA500')
ax.set_xlabel("Hour of Day (Lagos Time)")
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
if is_evening_forecast:
    st.info("🌙 It is evening now in Abuja. Today's sun is finished. Save your battery for tonight, use grid / generator if needed. Tomorrow's sun will charge again.")
elif user_type == "Grid Operator":
    if avg_power > 1.4:
        st.success("✅ High solar in Abuja. You can reduce reliance on grid / diesel today.")
    elif avg_power > 0.7:
        st.warning("⚠️ Moderate solar in Abuja. Keep backup ready, use solar + grid mix.")
    else:
        st.error("🔴 Low solar in Abuja. Expect cloudy / rainy. Switch to grid / battery backup.")
elif user_type == "Mini-Grid Operator":
    if avg_power > 1.4:
        st.success("✅ High solar in Abuja. Charge batteries fully, you can serve more households today.")
    elif avg_power > 0.7:
        st.warning("⚠️ Moderate solar in Abuja. Manage load, prioritize essential loads in evening.")
    else:
        st.error("🔴 Low solar in Abuja. Conserve battery, use diesel backup if available.")
else:
    if avg_power > 1.4:
        st.success("✅ Great day in Abuja! Run washing machine, pump water, charge devices with solar.")
    elif avg_power > 0.7:
        st.warning("⚠️ Okay day in Abuja. Use solar in afternoon, save battery for night.")
    else:
        st.error("🔴 Cloudy day in Abuja. Don't rely on solar alone, keep PHCN / generator ready.")

# --- BATTERY ADVICE ---
st.subheader("🔋 Battery Advice - What To Do Now")
now_hour = lagos_hour()
remaining_energy = float(hourly_forecast[now_hour:].sum())
battery_now_kwh = battery_size * (battery_now_pct / 100.0)
battery_needed_kwh = battery_size - battery_now_kwh
solar_after_load = max(0, remaining_energy - daily_load)
if is_evening_forecast:
    hours_to_full = 99
else:
    hours_to_full = (battery_needed_kwh / max(0.1, pv_now)) if pv_now > 0 else 99

b1, b2, b3 = st.columns(3)
b1.metric("🔋 Battery Now", f"{battery_now_kwh:.1f} / {battery_size:.1f} kWh")
b1.caption(f"{battery_now_pct}% full")
b2.metric("☀️ Solar Left", f"{remaining_energy:.1f} kWh")
b2.caption("Total sun left today, before your load")
b3.metric("⏱️ Time to Full", f"{hours_to_full:.1f} hrs" if hours_to_full < 20 else "Not today")

if is_evening_forecast:
    st.error(f"🔴 SAVE BATTERY FOR TONIGHT - Sun has set. No more charging today. You have {battery_now_kwh:.1f} kWh in battery. Use only lights / fan / phone. Use grid / generator if you must.")
elif pv_now > 2.0:
    st.success(f"✅ CHARGE NOW - Best window 11am to 3pm. You have ~{solar_after_load:.1f} kWh left after your load. Battery can be full in ~{hours_to_full:.1f} hrs. Run freezer / washing machine / pump now.")
elif pv_now > 0.8:
    st.warning(f"⚠️ CHARGE SLOWLY - Moderate sun. You have ~{solar_after_load:.1f} kWh left. Charge battery slowly, run only essential loads. Save battery for evening.")
else:
    st.error(f"🔴 SAVE BATTERY - Low sun. Only ~{solar_after_load:.1f} kWh left. Don't charge heavy loads. Keep battery for lights / fan / phone at night. Use grid / generator if you must.")

if solar_after_load >= battery_needed_kwh:
    st.info(f"💡 Good news: Today's remaining sun ({solar_after_load:.1f} kWh) is enough to fill your battery ({battery_needed_kwh:.1f} kWh needed).")
else:
    st.info(f"💡 Heads up: Today's remaining sun ({solar_after_load:.1f} kWh) is NOT enough to fill your battery ({battery_needed_kwh:.1f} kWh needed). Prioritize essentials.")

# --- BATTERY HEALTH ---
st.subheader("🩺 Battery Health - How Strong Is Your Battery?")
health_loss = (battery_age_months * 0.25) + (battery_cycles * 0.02)
health_pct = max(50.0, 100.0 - health_loss)
real_capacity = battery_size * (health_pct / 100.0)
lost_capacity = battery_size - real_capacity

h1, h2, h3 = st.columns(3)
h1.metric("🩺 Health", f"{health_pct:.0f}%")
h2.metric("🔋 Real Capacity", f"{real_capacity:.2f} kWh")
h2.caption(f"Was {battery_size:.1f} kWh when new")
h3.metric("📉 Lost", f"{lost_capacity:.2f} kWh")
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