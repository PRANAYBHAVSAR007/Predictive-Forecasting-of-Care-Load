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

    df.columns = df.columns.str.strip()

    # Date handling
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)

    # Convert to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.select_dtypes(include=[np.number])

    # Fill missing safely
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill()

    return df

df = load_data()

# =========================
# TARGET
# =========================
target_cols = [c for c in df.columns if "hhs" in c.lower()]

if not target_cols:
    st.error("❌ 'Children in HHS Care' column not found")
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

# Use fewer lags (LESS DATA LOSS)
df["lag_1"] = df[target].shift(1)
df["lag_7"] = df[target].shift(7)

# Rolling
df["rolling_mean_7"] = df[target].rolling(7).mean()

# Fill missing instead of dropping
df = df.replace([np.inf, -np.inf], np.nan)
df = df.ffill().bfill()

# =========================
# TRAIN
# =========================
features = ["lag_1", "lag_7", "rolling_mean_7", "Net Pressure"]

X = df[features]
y = df[target]

# FINAL CLEAN (no drop)
X = X.replace([np.inf, -np.inf], np.nan).ffill().bfill()
y = y.replace([np.inf, -np.inf], np.nan).ffill().bfill()

X = X.values
y = y.values

# Train model
model = RandomForestRegressor()
model.fit(X, y)

# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Controls")
forecast_days = st.sidebar.slider("Forecast Days", 3, 14, 7)

# =========================
# FORECAST
# =========================
forecast = []
current = df.iloc[-1:].copy()

current = current.ffill().bfill()

for i in range(forecast_days):
    X_pred = current[features].values
    pred = model.predict(X_pred)[0]
    forecast.append(pred)

    current["lag_1"] = pred
    current["lag_7"] = pred

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
