import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Safe ARIMA import
try:
    from statsmodels.tsa.arima.model import ARIMA
    arima_available = True
except:
    arima_available = False

from sklearn.ensemble import RandomForestRegressor

st.set_page_config(layout="wide")
st.title("📊 Predictive Forecasting of Care Load & Placement Demand")

@st.cache_data
def load_data():
    df = pd.read_csv("HHS_Unaccompanied_Alien_Children_Program.csv")

    # Clean column names
    df.columns = df.columns.str.strip()

    # Convert Date
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)

    # FORCE ALL NUMERIC COLUMNS TO FLOAT (CRITICAL FIX)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')

    # Convert only numeric-looking columns properly
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].astype(float)

    # NOW interpolation will work
    df[numeric_cols] = df[numeric_cols].interpolate()

    # Fill missing safely
    df = df.bfill().ffill()

    return df

df = load_data()

# TARGET COLUMN DETECTION (SAFE)
target = None
for col in df.columns:
    if "hhs care" in col.lower():
        target = col

if target is None:
    st.error("❌ Could not find 'Children in HHS Care' column")
    st.stop()

# CREATE STANDARD NAME
df.rename(columns={target: "Children in HHS Care"}, inplace=True)
target = "Children in HHS Care"

# FEATURE ENGINEERING
if "Children transferred out of CBP custody" in df.columns and "Children discharged from HHS Care" in df.columns:
    df["Net Pressure"] = df["Children transferred out of CBP custody"] - df["Children discharged from HHS Care"]
else:
    df["Net Pressure"] = 0

# Lag features
for lag in [1,7,14]:
    df[f"lag_{lag}"] = df[target].shift(lag)

df["rolling_mean_7"] = df[target].rolling(7).mean()
df["rolling_mean_14"] = df[target].rolling(14).mean()

df = df.dropna()

# TRAIN SPLIT
train_size = int(len(df)*0.8)
train = df[:train_size]

# SIDEBAR
st.sidebar.header("⚙️ Controls")

model_choice = st.sidebar.selectbox(
    "Model",
    ["Random Forest","ARIMA"] if arima_available else ["Random Forest"]
)

forecast_days = st.sidebar.slider("Forecast Days", 3, 14, 7)

forecast = []

# MODEL
if model_choice == "ARIMA" and arima_available:
    model = ARIMA(train[target], order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=forecast_days)

else:
    features = ["lag_1","lag_7","lag_14","rolling_mean_7","rolling_mean_14","Net Pressure"]

    X = train[features]
    y = train[target]

    model = RandomForestRegressor()
    model.fit(X, y)

    current = df.iloc[-1:].copy()

    for i in range(forecast_days):
        pred = model.predict(current[features])[0]
        forecast.append(pred)

        current["lag_1"] = pred
        current["lag_7"] = pred
        current["lag_14"] = pred

# PLOT
st.subheader("📈 Forecast")

fig, ax = plt.subplots(figsize=(12,5))
ax.plot(df[target], label="Historical")

future_dates = pd.date_range(start=df.index[-1], periods=forecast_days+1)[1:]
ax.plot(future_dates, forecast, color="red", label="Forecast")

ax.legend()
st.pyplot(fig)

# OPTIONAL CHARTS
if "Children discharged from HHS Care" in df.columns:
    st.subheader("📉 Discharge Trend")
    st.line_chart(df["Children discharged from HHS Care"])

st.subheader("⚠️ System Pressure")
st.line_chart(df["Net Pressure"])

# KPIs
st.subheader("📌 KPIs")
col1, col2 = st.columns(2)

col1.metric("Latest Care Load", int(df[target].iloc[-1]))
col2.metric("Net Pressure", int(df["Net Pressure"].iloc[-1]))

# Insights
st.subheader("🧠 Insights")
st.write("""
- Forecast enables proactive planning  
- Net pressure shows system stress  
- Model predicts short-term demand  
""")
