import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load your pre-trained model
model = load_model('models/stocks_model.keras')

# Streamlit UI
st.header('📈 Stock Market Predictor')

# Stock symbol input
stock = st.text_input('Enter Stock Symbol', 'GOOG')

# Date range for historical data
start = '2012-01-01'
end = '2025-05-01'

# Load stock data
data = yf.download(stock, start=start, end=end)

# Display basic moving averages
st.subheader('Price vs MA50')
ma_50_days = data['Close'].rolling(50).mean()
fig1 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r', label='MA50')
plt.plot(data['Close'], 'g', label='Closing Price')
plt.legend()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data['Close'].rolling(100).mean()
fig2 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r', label='MA50')
plt.plot(ma_100_days, 'b', label='MA100')
plt.plot(data['Close'], 'g', label='Closing Price')
plt.legend()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = data['Close'].rolling(200).mean()
fig3 = plt.figure(figsize=(8, 6))
plt.plot(ma_100_days, 'r', label='MA100')
plt.plot(ma_200_days, 'b', label='MA200')
plt.plot(data['Close'], 'g', label='Closing Price')
plt.legend()
st.pyplot(fig3)

# Data preprocessing for prediction
scaler = MinMaxScaler(feature_range=(0, 1))

# Use last 120 days of data, but pad 100 for prediction sequences
data_test = data.tail(120)
pas_100_days = data.tail(100)

# Combine and scale
data_test_full = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scaled = scaler.fit_transform(data_test_full[['Close']])

# Prepare input sequences for model
x = []
y = []
for i in range(100, data_test_scaled.shape[0]):
    x.append(data_test_scaled[i - 100:i])
    y.append(data_test_scaled[i, 0])

x, y = np.array(x), np.array(y)

# Predict
predicted = model.predict(x)

# Inverse scale
scale = 1 / scaler.scale_[0]
predict = predicted * scale
y = y * scale

# Plot predictions
st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8, 6))
plt.plot(y, 'r', label='Original Price')  
plt.plot(predict, 'g', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig4)
