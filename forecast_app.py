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

    # ✅ FIX DATE PROPERLY
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])   # 🔥 REMOVE NaT VALUES

    df = df.sort_values("Date")
    df.set_index("Date", inplace=True)

    # 🔥 FIX NUMBERS WITH COMMAS
    df["Children in HHS Care"] = df["Children in HHS Care"].astype(str).str.replace(",", "")
    df["Children in HHS Care"] = pd.to_numeric(df["Children in HHS Care"], errors="coerce")

    # Convert other columns
    for col in df.columns:
        if col != "Children in HHS Care":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clean values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill()

    return df

df = load_data()

# =========================
# TARGET
# =========================
target = "Children in HHS Care"

# =========================
# FEATURE
# =========================
df["lag_1"] = df[target].shift(1)
df = df.ffill().bfill()

# =========================
# TRAIN
# =========================
X = df[["lag_1"]].values
y = df[target].values

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

for i in range(forecast_days):
    pred = model.predict(current[["lag_1"]].values)[0]
    forecast.append(pred)
    current["lag_1"] = pred

# =========================
# PLOT
# =========================
st.subheader("📈 Forecast")

fig, ax = plt.subplots(figsize=(12,5))
ax.plot(df[target], label="Historical")

# ✅ SAFE DATE GENERATION
last_date = df.index[-1]

if pd.isna(last_date):
    last_date = pd.Timestamp.today()

future_dates = pd.date_range(start=last_date, periods=forecast_days+1)[1:]

ax.plot(future_dates, forecast, color="red", label="Forecast")

ax.legend()
st.pyplot(fig)

# =========================
# KPI
# =========================
st.subheader("📌 KPI")
st.metric("Latest Care Load", int(df[target].iloc[-1]))

# =========================
# INSIGHTS
# =========================
st.subheader("🧠 Insights")

st.write("""
- Cleaned dataset (fixed commas & dates)
- Model predicts based on recent trend
- Stable short-term forecasting
""")
