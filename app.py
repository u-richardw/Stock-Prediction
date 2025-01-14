import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
Sequential = tf.keras.models.Sequential
load_model = tf.keras.models.load_model
LSTM = tf.keras.layers.LSTM
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Tkinter issues
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pickle
from flask import Flask, request, jsonify, render_template
import os

app = Flask(__name__)

# Step 1: Fetch Historical Stock Data
def fetch_stock_data(ticker, start_date="2020-01-01"):
    try:
        end_date = datetime.today().strftime('%Y-%m-%d')  # Get today's date
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        if stock_data.empty:
            raise ValueError("No data found for ticker.")
        return stock_data['Close'].values.reshape(-1, 1), stock_data.index
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return None, None

# Step 2: Preprocess Data
def preprocess_data(data, sequence_length=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    x_train, y_train = [], []
    for i in range(sequence_length, len(scaled_data)):
        x_train.append(scaled_data[i-sequence_length:i, 0])
        y_train.append(scaled_data[i, 0])
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    return x_train, y_train, scaler

# Step 3: Build LSTM Model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Step 4: Train and Evaluate Model
def train_model(ticker):
    data, _ = fetch_stock_data(ticker)
    if data is None:
        return None, None
    
    x_train, y_train, scaler = preprocess_data(data)
    
    model = build_lstm_model((x_train.shape[1], 1))
    model.fit(x_train, y_train, epochs=30, batch_size=32)
    
    # Save model and scaler
    model.save(f'{ticker}_lstm_model.h5')
    with open(f'{ticker}_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    return model, scaler

# Step 5: Test Model Predictions
def test_model(ticker, future_days=30):
    if not os.path.exists(f'{ticker}_lstm_model.h5'):
        print(f"Model for {ticker} not found. Training now...")
        model, scaler = train_model(ticker)
        if model is None:
            return None
    else:
        model = load_model(f'{ticker}_lstm_model.h5')
        with open(f'{ticker}_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    
    data, dates = fetch_stock_data(ticker)
    if data is None:
        return None
    
    scaled_data = scaler.transform(data)
    sequence_length = 60
    
    x_input = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)
    future_predictions = []
    
    for _ in range(future_days):
        pred = model.predict(x_input, verbose=0)[0][0]
        future_predictions.append(pred)
        x_input = np.append(x_input[:, 1:, :], [[[pred]]], axis=1)
    
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    # Generate future dates for prediction
    future_dates = pd.date_range(start=dates[-1], periods=future_days+1, freq='D')[1:]

    # Improve X-axis and Y-axis accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(dates, data, label="Actual Price", color='blue')
    plt.plot(future_dates, future_predictions, label="Predicted Future Price", linestyle="dashed", color='orange')

    # X-axis: Date formatting
    plt.xlabel("Date")
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Show every 3 months
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # Y-axis: Stock Price
    plt.ylabel("Stock Price")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tick_params(axis='y', which='both', direction='in', length=6)

    plt.legend()
    plt.title(f"{ticker} Future Stock Price Prediction")

    # Ensure "static/" exists
    if not os.path.exists("static"):
        os.makedirs("static")

    plt.savefig("static/prediction.png")
    plt.close()
    
    return future_predictions.tolist()

# Step 6: Create Flask API and Web Dashboard
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    ticker = request.args.get('ticker', 'AAPL')
    future_days = int(request.args.get('days', 30))
    predictions = test_model(ticker, future_days)
    if predictions is None:
        return jsonify({"error": "Invalid ticker or no data available"}), 400
    return render_template('index.html', ticker=ticker, predictions=predictions, image_url='static/prediction.png')

if __name__ == "__main__":
    app.run(debug=True)
