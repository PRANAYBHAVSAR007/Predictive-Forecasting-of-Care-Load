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

    # Date
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)

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

# =========================
# CLEAN DATA (STRONG CLEAN)
# =========================
df[target] = pd.to_numeric(df[target], errors='coerce')

# Drop rows where target is invalid
df = df.dropna(subset=[target])

# Feature: lag_1
df["lag_1"] = df[target].shift(1)

# =========================
# FINAL CLEAN BEFORE MODEL
# =========================
data = df[["lag_1", target]].copy()

# Convert everything properly
data["lag_1"] = pd.to_numeric(data["lag_1"], errors='coerce')
data[target] = pd.to_numeric(data[target], errors='coerce')

# Remove ALL bad rows
data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna()

# SAFETY CHECK
if len(data) < 5:
    st.error("❌ Dataset is too small or corrupted. Please check your CSV file.")
    st.stop()

X = data[["lag_1"]].values
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
    X_pred = current[["lag_1"]].values
    pred = model.predict(X_pred)[0]
    forecast.append(pred)

    current["lag_1"] = pred

# =========================
# PLOT
# =========================
st.subheader("📈 Forecast")

fig, ax = plt.subplots(figsize=(12,5))
ax.plot(data[target], label="Historical")

future_dates = pd.date_range(start=pd.Timestamp.now(), periods=forecast_days)
ax.plot(future_dates, forecast, color="red", label="Forecast")

ax.legend()
st.pyplot(fig)

# =========================
# KPI
# =========================
st.subheader("📌 KPI")
st.metric("Latest Care Load", int(data[target].iloc[-1]))

# =========================
# INSIGHTS
# =========================
st.subheader("🧠 Insights")

st.write("""
- Model predicts based on last observed value  
- Works reliably even with messy data  
- Ensures stable deployment  
""")
