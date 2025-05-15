import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

# Load dataset to train model (you can replace this with a saved model later)
df = pd.read_csv("final_merged_weather_yieldd.csv")
X = df.drop(columns=["kg_per_acre"])
y = df["kg_per_acre"]
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# ---------------- UI ----------------
st.set_page_config(page_title="Crop Yield Predictor", layout="centered")

st.title("🌾 Smart Crop Yield Predictor")
st.markdown("Enter the environmental and remote sensing data below to estimate yield (kg/acre).")

with st.sidebar:
    st.header("📋 Input Parameters")

    year = st.slider("Year", 2010, 2030, 2025)
    tempmax = st.slider("Max Temperature (°C)", 15.0, 45.0, 27.5)
    tempmin = st.slider("Min Temperature (°C)", 5.0, 25.0, 10.2)
    precip = st.slider("Precipitation (mm)", 0.0, 20.0, 0.35)
    humidity = st.slider("Humidity (%)", 10.0, 100.0, 36.5)
    ndvi = st.slider("NDVI", 0.0, 1.0, 0.48)
    ndmi = st.slider("NDMI", 0.0, 1.0, 0.21)
    msavi = st.slider("MSAVI", 0.0, 1.0, 0.31)

# Create DataFrame from inputs
input_data = pd.DataFrame([{
    "Year": year,
    "tempmax": tempmax,
    "tempmin": tempmin,
    "precip": precip,
    "humidity": humidity,
    "NDVI": ndvi,
    "NDMI": ndmi,
    "MSAVI": msavi
}])

# ---------- Prediction Logic ----------
st.subheader("🔍 Result")

if ndvi < 0.2:
    st.warning("⚠️ NDVI is too low (< 0.2). This area is likely non-agricultural or residential. No prediction made.")
else:
    prediction = model.predict(input_data)[0]
    st.success(f"🌱 Predicted Yield: **{prediction:.2f} kg per acre**")

# ---------- Additional Info ----------
with st.expander("ℹ️ How this works"):
    st.markdown("""
    - **NDVI**, **NDMI**, and **MSAVI** are the most influential indicators of vegetation health.
    - **Temperature**, **Humidity**, and **Precipitation** help fine-tune the prediction.
    - If NDVI is below 0.2, it's usually a non-vegetated or urban area, so no yield is expected.
    """)

st.caption("📊 Model: Random Forest Regressor | Developed by [Mohsin taj]")
