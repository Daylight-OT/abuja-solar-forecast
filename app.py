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

# --- 24H CURVE ---
hours = np.arange(24)
solar_curve = np.maximum(0, np.sin((hours - 6) / 12 * np.pi)) ** 1.5
hourly_forecast = np.round(solar_curve * pred_pv * 1.8, 2)
avg_power = hourly_forecast.mean()

# --- STATUS FOR TODAY ---
last = df.iloc[-1]
irr = last['irradiance']
if irr >= 5:
    status = "Excellent ☀️"
elif irr >= 3:
    status = "Good 🌤️"
elif irr >= 1.5:
    status = "Fair ⛅"
else:
    status = "Low 🌧️"

# --- METRICS ---
st.subheader(f"📊 Live Forecast for Abuja - {datetime.now().strftime('%d %b %Y')}")
col1, col2, col3, col4 = st.columns(4)
col1.metric("☀️ Irradiance Today", f"{last['irradiance']:.2f} kWh/m²/day")
col2.metric("🌡️ Temperature", f"{last['temperature']:.1f} °C")
col3.metric("💧 Humidity", f"{last['humidity']:.1f} %")
col4.metric("⚡ Predicted PV Power", f"{pred_pv:.2f} kW", delta=status)

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

# --- DOWNLOAD ---
csv = hourly_df.to_csv(index=False)
st.download_button("⬇️ Download Forecast for Abuja (CSV)", csv, "solar_forecast_abuja.csv", "text/csv")
st.caption("Data: NASA POWER API (Live) | Models: Linear Regression vs XGBoost vs Random Forest | Location: Abuja, FCT (9.08, 7.49) | Seed: 42 for reproducibility")