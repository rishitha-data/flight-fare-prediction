import streamlit as st
import pandas as pd
import requests
import re
import os
import json
import time
from datetime import time as dt_time

# ================================
# CONFIG
# ================================
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
PREDICT_URL = f"{API_URL}/predict"
HEALTH_URL = f"{API_URL}/health"

DATA_PATH = os.path.join("data", "flight_data.csv")
METRICS_PATH = os.path.join("artifacts", "metrics.json")

st.set_page_config(
    page_title="Flight Fare Prediction System",
    page_icon="✈️",
    layout="wide"
)

# ================================
# LOAD DATA
# ================================
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_data
def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    return {}

df = load_data()
metrics = load_metrics()

# ================================
# CLEAN DATA
# ================================
for col in ["Airline", "Source", "Destination"]:
    df[col] = df[col].astype(str).str.strip()

airline_options = sorted(df["Airline"].unique())
source_options = sorted(df["Source"].unique())
destination_options = sorted(df["Destination"].unique())

# ================================
# API HEALTH CHECK
# ================================
def check_api():
    try:
        res = requests.get(HEALTH_URL, timeout=2)
        return res.status_code == 200
    except:
        return False

api_status = check_api()

# ================================
# HELPERS
# ================================
def duration_to_minutes(duration_str):
    try:
        h = re.search(r"(\d+)h", duration_str)
        m = re.search(r"(\d+)m", duration_str)

        hours = int(h.group(1)) if h else 0
        minutes = int(m.group(1)) if m else 0

        total = hours * 60 + minutes
        return total if total > 0 else None
    except:
        return None

# ================================
# HEADER
# ================================
st.title("✈️ Flight Fare Prediction System")

col1, col2 = st.columns([8, 1])
with col2:
    if api_status:
        st.success("API ✔")
    else:
        st.error("API ✖")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Records", len(df))
k2.metric("Avg Fare", f"₹{int(df['Price'].mean())}")
k3.metric("Max Fare", f"₹{int(df['Price'].max())}")
k4.metric("R²", metrics.get("R2", "N/A"))

st.divider()

# ================================
# INPUT SECTION
# ================================
c1, c2, c3 = st.columns(3)

with c1:
    airline = st.selectbox("Airline", airline_options)
    source = st.selectbox("Source", source_options)

    # 🔥 Dynamic destination filtering
    destination_filtered = [d for d in destination_options if d != source]
    destination = st.selectbox("Destination", destination_filtered)

with c2:
    stops = st.slider("Stops", 0, 4, 1)
    date = st.date_input("Journey Date")

    day = date.day
    month = date.month
    is_weekend = int(date.weekday() >= 5)

with c3:
    # 🔥 Smart default duration from dataset
    default_duration = "2h 30m"
    duration_str = st.text_input("Duration (e.g. 2h 50m)", default_duration)
    duration = duration_to_minutes(duration_str)

    dep_time = st.time_input("Departure", dt_time(10, 0), key="dep_time")
    arr_time = st.time_input("Arrival", dt_time(13, 0), key="arr_time")

dep_hour, dep_min = dep_time.hour, dep_time.minute
arr_hour, arr_min = arr_time.hour, arr_time.minute
is_peak = int(6 <= dep_hour <= 9 or 18 <= dep_hour <= 21)

st.divider()

# ================================
# VALIDATION
# ================================
errors = []

if duration is None:
    errors.append("Invalid duration format (use like '2h 30m')")

if not api_status:
    errors.append("Backend API is not running")

if errors:
    for err in errors:
        st.warning(err)

# ================================
# TRIP SUMMARY
# ================================
st.markdown(f"""
### ✈️ Trip Summary  
**{source} → {destination}** | {airline} | {stops} stop(s)  
Departure: {dep_hour:02d}:{dep_min:02d} | Arrival: {arr_hour:02d}:{arr_min:02d}
""")

# ================================
# ACTION BUTTONS
# ================================
col_btn1, col_btn2 = st.columns([1, 1])

with col_btn1:
    predict_btn = st.button("Predict 💰", disabled=len(errors) > 0)

with col_btn2:
    if st.button("Reset 🔄"):
        st.rerun()

# ================================
# PREDICTION
# ================================
if predict_btn:

    input_data = {
        "Airline": airline,
        "Source": source,
        "Destination": destination,
        "Total_Stops": stops,
        "Duration_Minutes": duration,
        "Journey_Day": day,
        "Journey_Month": month,
        "Is_Weekend": is_weekend,
        "Dep_Hour": dep_hour,
        "Dep_Min": dep_min,
        "Arr_Hour": arr_hour,
        "Arr_Min": arr_min,
        "Is_Peak_Departure": is_peak
    }

    try:
        with st.spinner("Predicting fare... ✈️"):
            start = time.time()
            response = requests.post(PREDICT_URL, json=input_data, timeout=5)
            end = time.time()

        if response.status_code != 200:
            st.error("Backend unavailable. Please try again later.")
            st.stop()

        result = response.json()

        if "error" in result:
            st.error(result["error"])
            st.stop()

        price = result["predicted_price"]

        st.markdown("## 💰 Predicted Fare")
        st.success(f"### ₹ {price:,}")

        lower = int(price * 0.85)
        upper = int(price * 1.15)

        st.info(f"Estimated Range: ₹{lower:,} – ₹{upper:,}")

        # 🔥 API latency
        st.caption(f"⏱ Response time: {round(end - start, 2)} sec")

        st.caption("💡 Prices increase during peak hours, weekends, and higher stops.")

    except requests.exceptions.Timeout:
        st.error("Request timed out. Try again.")
    except Exception as e:
        st.error(f"Error: {e}")

# ================================
# DATA PREVIEW
# ================================
st.subheader("📊 Dataset Preview")
st.dataframe(df.head(100), use_container_width=True)

# ================================
# FOOTER
# ================================
st.markdown("---")
st.caption("Built with FastAPI + Streamlit | ML Model: XGBoost")