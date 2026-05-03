import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Safe import for ARIMA
try:
    from statsmodels.tsa.arima.model import ARIMA
    arima_available = True
except:
    arima_available = False

from sklearn.ensemble import RandomForestRegressor

# PAGE CONFIG
st.set_page_config(layout="wide")

st.title("📊 Predictive Forecasting of Care Load & Placement Demand")

# LOAD DATA
@st.cache_data
def load_data():
    df = pd.read_csv("HHS_Unaccompanied_Alien_Children_Program.csv")

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)

    # FIX: Interpolate ONLY numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate()

    # Fill remaining missing values
    df = df.bfill()
    df = df.ffill()

    return df

df = load_data()

target = 'Children in HHS Care'

# FEATURE ENGINEERING
df['Net Pressure'] = df['Children transferred out of CBP custody'] - df['Children discharged from HHS Care']

for lag in [1, 7, 14]:
    df[f'lag_{lag}'] = df[target].shift(lag)

df['rolling_mean_7'] = df[target].rolling(7).mean()
df['rolling_mean_14'] = df[target].rolling(14).mean()

df = df.dropna()

# TRAIN TEST SPLIT
train_size = int(len(df) * 0.8)
train = df[:train_size]
test = df[train_size:]

# SIDEBAR CONTROLS
st.sidebar.header("⚙️ Controls")

model_options = ["Random Forest"]
if arima_available:
    model_options.insert(0, "ARIMA")

model_choice = st.sidebar.selectbox("Select Model", model_options)
forecast_days = st.sidebar.slider("Forecast Days", 3, 14, 7)

# FORECASTING
forecast = []

if model_choice == "ARIMA" and arima_available:
    model = ARIMA(train[target], order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=forecast_days)

else:
    features = ['lag_1','lag_7','lag_14','rolling_mean_7','rolling_mean_14','Net Pressure']

    X_train = train[features]
    y_train = train[target]

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    current = df.iloc[-1:].copy()

    for i in range(forecast_days):
        pred = model.predict(current[features])[0]
        forecast.append(pred)

        # update lag features dynamically
        current['lag_1'] = pred
        current['lag_7'] = current['lag_1']
        current['lag_14'] = current['lag_7']

# PLOT FORECAST
st.subheader("📈 Future Care Load Forecast")

fig, ax = plt.subplots(figsize=(12,5))

ax.plot(df[target], label="Historical Data")

future_dates = pd.date_range(start=df.index[-1], periods=forecast_days+1)[1:]

ax.plot(future_dates, forecast, label="Forecast", color='red')

ax.legend()
ax.set_title("Care Load Forecast")

st.pyplot(fig)

# DISCHARGE DEMAND
st.subheader("📉 Discharge Demand Trend")
st.line_chart(df['Children discharged from HHS Care'])

# SYSTEM PRESSURE
st.subheader("⚠️ System Pressure Indicator")
st.line_chart(df['Net Pressure'])

# KPIs
st.subheader("📌 Key Indicators")

col1, col2 = st.columns(2)

col1.metric("Latest Care Load", int(df[target].iloc[-1]))
col2.metric("Latest Net Pressure", int(df['Net Pressure'].iloc[-1]))

# INSIGHTS
st.subheader("🧠 Insights")

st.write("""
- Positive Net Pressure indicates increasing system stress  
- Forecast helps predict overcrowding risk  
- Random Forest captures complex relationships  
- ARIMA captures trend patterns (if enabled)  
""")
