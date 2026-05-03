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

    # Convert numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.select_dtypes(include=[np.number])

    # Clean values
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

# Drop rows where target is missing
df = df.dropna(subset=[target])

# =========================
# FEATURE ENGINEERING (MINIMAL)
# =========================
df["lag_1"] = df[target].shift(1)

# Fill missing values instead of dropping
df = df.ffill().bfill()

# =========================
# TRAIN
# =========================
features = ["lag_1"]

X = df[features].values
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
    X_pred = current[features].values
    pred = model.predict(X_pred)[0]
    forecast.append(pred)

    current["lag_1"] = pred

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
# KPI
# =========================
st.subheader("📌 KPI")

st.metric("Latest Care Load", int(df[target].iloc[-1]))

# =========================
# INSIGHTS
# =========================
st.subheader("🧠 Insights")

st.write("""
- Model uses previous day trend to predict next values  
- Works even with limited data  
- Provides stable short-term forecasts  
""")
