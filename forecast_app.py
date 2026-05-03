import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(layout="wide")
st.title("📊 Predictive Forecasting of Care Load")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("HHS_Unaccompanied_Alien_Children_Program.csv")

    # Clean column names
    df.columns = df.columns.str.strip()

    # Convert Date
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)

    # Convert all columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Keep only numeric columns
    df = df.select_dtypes(include=[np.number])

    # Replace infinite and fill missing
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill()

    return df

df = load_data()

# =========================
# TARGET COLUMN
# =========================
target_cols = [c for c in df.columns if "hhs" in c.lower()]

if not target_cols:
    st.error("❌ 'Children in HHS Care' column not found")
    st.stop()

target = target_cols[0]
df.rename(columns={target: "Children in HHS Care"}, inplace=True)
target = "Children in HHS Care"

# =========================
# FEATURE ENGINEERING
# =========================
cols = df.columns.tolist()

if len(cols) >= 3:
    df["Net Pressure"] = df[cols[1]] - df[cols[2]]
else:
    df["Net Pressure"] = 0

# Lag features
for lag in [1, 7, 14]:
    df[f"lag_{lag}"] = df[target].shift(lag)

# Rolling feature
df["rolling_mean_7"] = df[target].rolling(7).mean()

# Fill after feature creation
df = df.replace([np.inf, -np.inf], np.nan)
df = df.ffill().bfill()

# =========================
# TRAIN SPLIT
# =========================
train_size = int(len(df) * 0.8)
train = df[:train_size]

# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Controls")
forecast_days = st.sidebar.slider("Forecast Days", 3, 14, 7)

# =========================
# MODEL
# =========================
features = ["lag_1", "lag_7", "lag_14", "rolling_mean_7", "Net Pressure"]

# 🔥 FINAL CLEAN (MOST IMPORTANT STEP)
data = train[features + [target]].copy()

data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna()

if len(data) < 10:
    st.error("❌ Not enough clean data. Please check dataset.")
    st.stop()

X = data[features].values
y = data[target].values

model = RandomForestRegressor()
model.fit(X, y)

# =========================
# FORECAST
# =========================
forecast = []
current = df.iloc[-1:].copy()

current = current.replace([np.inf, -np.inf], np.nan)
current = current.ffill().bfill()

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
ax.plot(df[target], label="Historical")

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
st.subheader("📌 KPIs")

col1, col2 = st.columns(2)
col1.metric("Latest Care Load", int(df[target].iloc[-1]))
col2.metric("Net Pressure", int(df["Net Pressure"].iloc[-1]))

# =========================
# INSIGHTS
# =========================
st.subheader("🧠 Insights")

st.write("""
- Forecast predicts short-term care demand  
- Net pressure indicates system stress  
- Enables proactive planning  
""")
