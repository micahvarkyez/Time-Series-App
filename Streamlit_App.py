import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime
import numpy as np
from io import BytesIO

# Title
st.title('Time Series Forecaster')

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
    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)
    st.write("Cleaned Data Preview:", data.head())

    # Convert date column to time series
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    # Time Series Decomposition
    st.subheader('Time Series Decomposition')
    column = 'Adj Close'  # Typically, we forecast based on the adjusted closing price
    result = seasonal_decompose(data[column], model='multiplicative', period=12)
            
    # Plot the decomposed components
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    result.observed.plot(ax=ax1)
    ax1.set_ylabel('Observed')
    result.trend.plot(ax=ax2)
    ax2.set_ylabel('Trend')
    result.seasonal.plot(ax=ax3)
    ax3.set_ylabel('Seasonal')
    result.resid.plot(ax=ax4)
    ax4.set_ylabel('Residual')
    plt.xlabel('Date')
    plt.tight_layout()
    st.pyplot(fig)

    # Forecasting with SARIMA
    st.subheader('Forecasting with SARIMA')
    model = SARIMAX(data[column], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit(disp=False)
       
    # Model summary
    st.subheader('Model Summary')
    st.text(model_fit.summary())
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

    # Save plot as image or PDF
    img_buffer = BytesIO()
    pdf_buffer = BytesIO()
    fig.savefig(img_buffer, format='png')
    fig.savefig(pdf_buffer, format='pdf')
    img_buffer.seek(0)
    pdf_buffer.seek(0)
    st.download_button(label='Download Forecast as Image', data=img_buffer, file_name='forecast.png', mime='image/png')
    st.download_button(label='Download Forecast as PDF', data=pdf_buffer, file_name='forecast.pdf', mime='application/pdf')
