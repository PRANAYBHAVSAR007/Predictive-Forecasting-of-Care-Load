import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

st.title("📊 Predictive Forecasting of Care Load & Placement Demand")

# LOAD DATA
df = pd.read_csv("HHS_Unaccompanied_Alien_Children_Program.csv")

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df.set_index('Date', inplace=True)

df = df.interpolate()

target = 'Children in HHS Care'

# FEATURE ENGINEERING
df['Net Pressure'] = df['Children transferred out of CBP custody'] - df['Children discharged from HHS Care']

for lag in [1,7,14]:
    df[f'lag_{lag}'] = df[target].shift(lag)

df['rolling_mean_7'] = df[target].rolling(7).mean()
df['rolling_mean_14'] = df[target].rolling(14).mean()

df = df.dropna()

# TRAIN TEST SPLIT
train_size = int(len(df)*0.8)
train = df[:train_size]
test = df[train_size:]

# SIDEBAR
st.sidebar.header("⚙️ Controls")

model_choice = st.sidebar.selectbox("Select Model", ["ARIMA", "Random Forest"])
forecast_days = st.sidebar.slider("Forecast Days", 3, 14, 7)

# MODEL LOGIC

if model_choice == "ARIMA":
    model = ARIMA(train[target], order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=forecast_days)

elif model_choice == "Random Forest":
    features = ['lag_1','lag_7','lag_14','rolling_mean_7','rolling_mean_14','Net Pressure']

    X_train = train[features]
    y_train = train[target]

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    last_data = df.iloc[-1:]

    forecast = []

    current = last_data.copy()

    for i in range(forecast_days):
        pred = model.predict(current[features])[0]
        forecast.append(pred)

        current['lag_1'] = pred

# PLOT

st.subheader("📈 Forecasted Care Load")

fig, ax = plt.subplots(figsize=(10,5))

ax.plot(df[target], label="Historical Data")

future_dates = pd.date_range(start=df.index[-1], periods=forecast_days+1)[1:]

ax.plot(future_dates, forecast, label="Forecast", color='red')

ax.legend()

st.pyplot(fig)

# DISCHARGE DEMAND

st.subheader("📉 Discharge Demand (Recent Trend)")

st.line_chart(df['Children discharged from HHS Care'])

# SYSTEM PRESSURE

st.subheader("⚠️ System Pressure Indicator")

st.line_chart(df['Net Pressure'])

# INSIGHTS PANEL

st.subheader("🧠 Insights")

st.write("""
- Positive Net Pressure indicates system stress  
- Forecast helps predict overcrowding risk  
- ARIMA captures trend; Random Forest captures patterns  
""")