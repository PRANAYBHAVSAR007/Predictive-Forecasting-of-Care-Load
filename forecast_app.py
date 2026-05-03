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

    # Date setup
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)

    # Detect columns safely
    col_map = {}

    for col in df.columns:
        c = col.lower()
        if "hhs care" in c:
            col_map["hhs"] = col
        elif "transferred" in c:
            col_map["transfer"] = col
        elif "discharged" in c:
            col_map["discharge"] = col

    # Convert numeric safely
    for key in col_map:
        df[col_map[key]] = pd.to_numeric(df[col_map[key]], errors='coerce')

    # Rename ONLY if present
    if "hhs" in col_map:
        df.rename(columns={col_map["hhs"]: "Children in HHS Care"}, inplace=True)

    if "transfer" in col_map:
        df.rename(columns={col_map["transfer"]: "Children transferred"}, inplace=True)

    if "discharge" in col_map:
        df.rename(columns={col_map["discharge"]: "Children discharged"}, inplace=True)

    # Fill missing numeric data
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate()

    df = df.bfill().ffill()

    return df

df = load_data()

# TARGET
target = "Children in HHS Care"

# SAFE CHECK
if target not in df.columns:
    st.error("❌ 'Children in HHS Care' column not found in dataset")
    st.stop()

# FEATURE ENGINEERING
if "Children transferred" in df.columns and "Children discharged" in df.columns:
    df["Net Pressure"] = df["Children transferred"] - df["Children discharged"]
else:
    df["Net Pressure"] = 0  # fallback

# Lag features
for lag in [1,7,14]:
    df[f"lag_{lag}"] = df[target].shift(lag)

df["rolling_mean_7"] = df[target].rolling(7).mean()
df["rolling_mean_14"] = df[target].rolling(14).mean()

df = df.dropna()

# Train-test split
train_size = int(len(df)*0.8)
train = df[:train_size]

# Sidebar
st.sidebar.header("⚙️ Controls")

model_choice = st.sidebar.selectbox("Model", ["Random Forest","ARIMA"] if arima_available else ["Random Forest"])
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

# Charts
if "Children discharged" in df.columns:
    st.subheader("📉 Discharge Trend")
    st.line_chart(df["Children discharged"])

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
- Net pressure indicates system stress  
- Model predicts short-term care load demand  
""")
