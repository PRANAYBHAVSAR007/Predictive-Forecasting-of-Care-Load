import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Optional ARIMA
try:
    from statsmodels.tsa.arima.model import ARIMA
    arima_available = True
except:
    arima_available = False

from sklearn.ensemble import RandomForestRegressor

st.set_page_config(layout="wide")
st.title("📊 Predictive Forecasting of Care Load & Placement Demand")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("HHS_Unaccompanied_Alien_Children_Program.csv")

    df.columns = df.columns.str.strip()

    # Date handling
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)

    # Convert numeric safely
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.select_dtypes(include=[np.number])

    # Fill missing values (IMPORTANT FIX)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(method='ffill')
    df = df.fillna(method='bfill')

    return df

df = load_data()

# =========================
# TARGET
# =========================
target_cols = [c for c in df.columns if "hhs" in c.lower()]

if not target_cols:
    st.error("❌ HHS Care column not found")
    st.stop()

target = target_cols[0]
df.rename(columns={target: "Children in HHS Care"}, inplace=True)
target = "Children in HHS Care"

# =========================
# FEATURES
# =========================
cols = df.columns.tolist()

if len(cols) >= 3:
    df["Net Pressure"] = df[cols[1]] - df[cols[2]]
else:
    df["Net Pressure"] = 0

# Lag features
for lag in [1, 7, 14]:
    df[f"lag_{lag}"] = df[target].shift(lag)

# Rolling features
df["rolling_mean_7"] = df[target].rolling(7).mean()
df["rolling_mean_14"] = df[target].rolling(14).mean()

# 🔥 DO NOT DROP DATA — FILL INSTEAD
df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(method='ffill')
df = df.fillna(method='bfill')

# =========================
# TRAIN SPLIT
# =========================
train_size = int(len(df) * 0.8)
train = df[:train_size]

# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Controls")

model_choice = st.sidebar.selectbox(
    "Model",
    ["Random Forest", "ARIMA"] if arima_available else ["Random Forest"]
)

forecast_days = st.sidebar.slider("Forecast Days", 3, 14, 7)

forecast = []

# =========================
# MODEL
# =========================
if model_choice == "ARIMA" and arima_available:
    try:
        model = ARIMA(train[target], order=(5,1,0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=forecast_days)
    except:
        st.warning("ARIMA failed → switching to Random Forest")
        model_choice = "Random Forest"

if model_choice == "Random Forest":

    features = ["lag_1","lag_7","lag_14","rolling_mean_7","rolling_mean_14","Net Pressure"]

    X = train[features]
    y = train[target]

    # 🔥 FINAL CLEAN (NO DROP, ONLY FILL)
    X = X.replace([np.inf, -np.inf], np.nan)
    y = y.replace([np.inf, -np.inf], np.nan)

    X = X.fillna(method='ffill').fillna(method='bfill')
    y = y.fillna(method='ffill').fillna(method='bfill')

    X = X.values
    y = y.values

    model = RandomForestRegressor()
    model.fit(X, y)

    current = df.iloc[-1:].copy()

    current = current.fillna(method='ffill').fillna(method='bfill')

    for i in range(forecast_days):
        X_pred = current[features].values
        pred = model.predict(X_pred)[0]
        forecast.append(pred)

        current["lag_1"] = pred
        current["lag_7"] = pred
        current["lag_14"] = pred

# =========================
# PLOT
# =========================
st.subheader("📈 Forecast")

fig, ax = plt.subplots(figsize=(12,5))
ax.plot(df[target], label="Historical Data")

future_dates = pd.date_range(start=df.index[-1], periods=forecast_days+1)[1:]
ax.plot(future_dates, forecast, color="red", label="Forecast")

ax.legend()
st.pyplot(fig)

# =========================
# VISUALS
# =========================
st.subheader("⚠️ System Pressure")
st.line_chart(df["Net Pressure"])

# =========================
# KPIs
# =========================
st.subheader("📌 Key Indicators")

col1, col2 = st.columns(2)
col1.metric("Latest Care Load", int(df[target].iloc[-1]))
col2.metric("Net Pressure", int(df["Net Pressure"].iloc[-1]))

# =========================
# INSIGHTS
# =========================
st.subheader("🧠 Insights")

st.write("""
- Forecast predicts future care demand  
- Net pressure indicates system stress  
- Helps proactive planning  
""")
