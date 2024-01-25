# loading required frameworks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from model import models

# creating the  title
st.title("Stock Trend Prediction")

# loading the model
predictor = models

# defining the required variables
start = st.text_input("Enter starting date: ", "2013-10-13")
end = st.text_input("Enter ending date: ", "2023-10-15")
ticker_symbol = st.text_input("Enter the stock ticker symbol: ", "GOOG")

# downloading the data
st.subheader("Stock Data for the Company")

# creating a selection box to select column from data
price = st.sidebar.selectbox("Pick the value you want to predict",
                             ("Close", "Open", "High", "Low"))

data = yf.download(tickers=ticker_symbol, start=start, end=end)
st.write(data.head())
st.write(data.tail())

# Statistics information of the data
st.subheader("Statistical Values of Stock Data")
st.write(data.describe())

# Adding visualizations
st.subheader(f"{price} Value chart vs Time")
fig = plt.figure(figsize=(12, 6))
plt.plot(data[price])
st.pyplot(fig)

st.subheader(f"{price} Value chart vs Time with 100 Days Moving Average")
# calculating rolling average for 50 days closing value
ma50_value = data[price].rolling(50).mean()
fig1 = plt.figure(figsize=(12, 6))
plt.plot(data[price], "r")
plt.plot(ma50_value, "b")
st.pyplot(fig1)

# splitting the data in training and testing i.e. 80% Training and 20% testing
data_train = pd.DataFrame(data[price][0:int(len(data)*0.8)])
data_test = pd.DataFrame(data[price][int(len(data)*0.8): int(len(data))])

# Creating the MinMaxScaler Object
mn = MinMaxScaler(feature_range=(0, 1))


# getting the last 60 days data to create window for the first value in the data_test variable
past_60_days = data_train.tail(60)

# concatenating the two dataframes
final_test = pd.concat([past_60_days, data_test], axis=0)

# scaling the data
final_test_scaled = mn.fit_transform(final_test)

# creating windows for testing phase
x_test = []
y_test = []
for i in range(60, final_test_scaled.shape[0]):
    x_test.append(final_test_scaled[i - 60: i])
    y_test.append(final_test_scaled[i, 0])

# converting them into arrays
x_test, y_test = np.array(x_test), np.array(y_test)

# Predictions
y_prediction = predictor.predict(x_test)
scaler = mn.scale_

# defining scale factor
scale_factor = 1 / scaler[0]

# converting the scaled values into original values/ price
y_test_actual = y_test * scale_factor
y_prediction_actual = y_prediction * scale_factor

# Final Chart : plotting the actual and predicted values on a line plot
st.header("Actual vs Predicted Chart")
fig2 = plt.figure(figsize=(12, 5))
plt.plot(y_test_actual, 'b', label="Original Price")
plt.plot(y_prediction_actual, 'r', label="Predicted Price")

# providing labels to both axis
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig2)
