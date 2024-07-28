pip install streamlit pandas yfinance statsmodels matplotlib

import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime

# Title
st.title('Sales Forecasting App')

# User Inputs
ticker = st.text_input('Enter Stock Ticker Symbol (e.g., AAPL)', 'AAPL')
forecast_horizon = st.number_input('Enter Forecast Horizon (months)', min_value=1, max_value=36, value=12)
start_date = st.date_input('Start Date', value=datetime(2010, 1, 1))
end_date = st.date_input('End Date', value=datetime.today())

# Fetch Data
if st.button('Fetch Data'):
    data = yf.download(ticker, start=start_date, end=end_date, interval='1mo')
    data.reset_index(inplace=True)
    st.write("Data Preview:", data.head())

    # Data Cleaning
    st.subheader('Data Cleaning')
    if st.button('Handle Missing Values and Duplicates'):
        data.dropna(inplace=True)
        data.drop_duplicates(inplace=True)
        st.write("Cleaned Data Preview:", data.head())

    # Convert date column to time series
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    # Time Series Decomposition
    st.subheader('Time Series Decomposition')
    column = 'Close'  # Typically, we forecast based on the closing price
    if st.button('Decompose Time Series'):
        result = seasonal_decompose(data[column], model='additive', period=12)
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))
        result.observed.plot(ax=ax1)
        result.trend.plot(ax=ax2)
        result.seasonal.plot(ax=ax3)
        result.resid.plot(ax=ax4)
        st.pyplot(fig)

    # Forecasting with SARIMA
    st.subheader('Forecasting with SARIMA')
    if st.button('Forecast'):
        model = SARIMAX(data[column], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        model_fit = model.fit(disp=False)
        forecast = model_fit.get_forecast(steps=forecast_horizon)
        forecast_ci = forecast.conf_int()

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 5))
        data[column].plot(ax=ax, label='Actual')
        forecast.predicted_mean.plot(ax=ax, label='Forecast')
        ax.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='k', alpha=.2)
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        plt.legend()
        st.pyplot(fig)
