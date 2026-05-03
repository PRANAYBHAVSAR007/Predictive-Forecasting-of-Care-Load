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

    # Convert numeric safely
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.select_dtypes(include=[np.number])

    # Remove infinite values
    df = df.replace([np.inf, -np.inf], np.nan)

    return df

df = load_data()

# =========================
# TARGET
# =========================
target_cols = [c for c in df.columns if "hhs" in c.lower()]

if not target_cols:
    st.error("❌ Target column not found")
    st.stop()

target = target_cols[0]
df.rename(columns={target: "Children in HHS Care"}, inplace=True)
target = "Children in HHS Care"

# Drop rows where target is NaN
df = df.dropna(subset=[target])

# =========================
# FEATURE ENGINEERING
# =========================
df["Net Pressure"] = df.iloc[:,1] - df.iloc[:,2] if len(df.columns) >= 3 else 0

df["lag_1"] = df[target].shift(1)
df["lag_7"] = df[target].shift(7)
df["rolling_mean_7"] = df[target].rolling(7).mean()

# =========================
# FINAL CLEAN (MOST IMPORTANT)
# =========================
features = ["lag_1", "lag_7", "rolling_mean_7", "Net Pressure"]

data = df[features + [target]].copy()

# Remove all invalid values in ONE step
data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna()

# Safety check
if len(data) < 20:
    st.error("❌ Dataset too small after cleaning. Please check your CSV.")
    st.stop()

X = data[features].values
y = data[target].values

# =========================
# MODEL
# =========================
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
current = data.iloc[-1:].copy()

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
ax.plot(data[target], label="Historical")

future_dates = pd.date_range(start=data.index[-1], periods=forecast_days+1)[1:]
ax.plot(future_dates, forecast, color="red", label="Forecast")

ax.legend()
st.pyplot(fig)

# =========================
# VISUALS
# =========================
st.subheader("⚠️ System Pressure")
st.line_chart(data["Net Pressure"])

# =========================
# KPIs
# =========================
st.subheader("📌 KPIs")

col1, col2 = st.columns(2)
col1.metric("Latest Care Load", int(data[target].iloc[-1]))
col2.metric("Net Pressure", int(data["Net Pressure"].iloc[-1]))

# =========================
# INSIGHTS
# =========================
st.subheader("🧠 Insights")

st.write("""
- Forecast predicts short-term care demand  
- Net pressure indicates system stress  
- Supports proactive planning  
""")
