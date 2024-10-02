import os
import pandas as pd
import numpy as np
import yfinance as yf
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__)

MODEL_DIR = 'saved_models'

# Ensure the directory exists to store the models and scalers
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Load the model and scaler
def load_model_and_scaler(stock_name):
    model_path = os.path.join(MODEL_DIR, f'{stock_name}_best_model.h5')
    scaler_path = os.path.join(MODEL_DIR, f'{stock_name}_scaler.npy')

    try:
        model = load_model(model_path)
        scaler = np.load(scaler_path, allow_pickle=True).item()
        return model, scaler
    except Exception as e:
        print(f"Error loading model or scaler for {stock_name}: {e}")
        return None, None

# Fetch Historical Data
def fetch_historical_data(stock_name):
    try:
        stock_data = yf.download(stock_name + '.NS', period='5y', interval='1d')
        return stock_data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# Prepare the data for predictions to match training features
def prepare_data_for_prediction(stock_data):
    stock_data['MA10'] = stock_data['Close'].rolling(window=10).mean()
    stock_data['MA50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data.fillna(method='backfill', inplace=True)

    features = stock_data[['Close', 'MA10', 'MA50']].values
    return features

# Predict the next day's price based on the last sequence
def predict_next_day_price(model, scaler, last_sequence):
    last_sequence_scaled = scaler.transform(last_sequence)
    X_test = last_sequence_scaled.reshape(1, last_sequence_scaled.shape[0], last_sequence_scaled.shape[1])

    predicted_price_scaled = model.predict(X_test)
    predicted_price = scaler.inverse_transform(
        np.concatenate(
            [predicted_price_scaled, np.zeros((predicted_price_scaled.shape[0], last_sequence_scaled.shape[1] - 1))],
            axis=1
        )
    )[:, 0]

    return predicted_price[0]

# Fetch today's actual price using yfinance
def fetch_actual_price(stock_name):
    # Fetch data for the current day (today's price)
    try:
        actual_data = yf.download(stock_name + '.NS', period='1d', interval='1m')  # Using '1m' interval for real-time data
        if not actual_data.empty:
            actual_price_today = actual_data['Close'].iloc[-1]  # Get the last available close price (most recent)
            return actual_price_today
        else:
            print(f"No data available for {stock_name} for today.")
            return None
    except Exception as e:
        print(f"Error fetching actual price for {stock_name}: {e}")
        return None

# Calculate accuracy based on actual and predicted price
def calculate_accuracy(predicted_price, actual_price):
    accuracy = (1 - abs((predicted_price - actual_price) / actual_price)) * 100
    return accuracy

# Predict the next 7 days
def predict_next_7_days(stock_name, model, scaler, stock_data):
    window_size = 120
    data = stock_data.copy()
    predicted_prices = []

    for _ in range(7):
        features = prepare_data_for_prediction(data)
        last_sequence = features[-window_size:]
        last_sequence_scaled = scaler.transform(last_sequence)
        X_test = last_sequence_scaled.reshape(1, last_sequence_scaled.shape[0], last_sequence_scaled.shape[1])

        predicted_price_scaled = model.predict(X_test)
        predicted_price = scaler.inverse_transform(
            np.concatenate(
                [predicted_price_scaled, np.zeros((1, features.shape[1] - 1))],
                axis=1
            )
        )[:, 0][0]

        predicted_prices.append(predicted_price)

        # Append the predicted price to the data for future predictions
        new_row = pd.DataFrame({'Close': [predicted_price]})
        data = pd.concat([data, new_row], ignore_index=True)

    # Use tomorrow's date as the start for future predictions
    tomorrow = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
    future_dates = pd.date_range(start=tomorrow, periods=7)

    # Create DataFrame for predicted prices and future dates
    prediction_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': predicted_prices})
    prediction_df.set_index('Date', inplace=True)

    print(f"Predicted future dates: {prediction_df.index}")
    return prediction_df

# List available stock models
def list_available_stocks():
    stock_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('_best_model.h5')]
    stock_names = [f.split('_best_model.h5')[0] for f in stock_files]
    return stock_names

@app.route('/')
def index():
    available_stocks = list_available_stocks()  # Get the available stock models
    return render_template('index.html', available_stocks=available_stocks)

@app.route('/predict', methods=['POST'])
def predict():
    stock_name = request.form['stock_name'].upper()
    model, scaler = load_model_and_scaler(stock_name)
    
    if model is None or scaler is None:
        return f"Error loading model for {stock_name}. Please ensure the model is trained."

    # Fetch historical data
    stock_data = fetch_historical_data(stock_name)
    if stock_data is None:
        return f"Error fetching data for {stock_name}. Please try again."

    # Prepare the data and predict today's price
    features = prepare_data_for_prediction(stock_data)
    last_sequence = features[-120:]
    predicted_price_today = predict_next_day_price(model, scaler, last_sequence)

    # Fetch today's actual price (real-time data)
    actual_price_today = fetch_actual_price(stock_name)

    if actual_price_today is None:
        return f"Error fetching today's actual price for {stock_name}. Please try again later."

    # Calculate accuracy
    accuracy_today = calculate_accuracy(predicted_price_today, actual_price_today)

    # Predict the next 7 days
    predicted_prices_df = predict_next_7_days(stock_name, model, scaler, stock_data)

    return render_template('prediction.html', stock_name=stock_name, today_price=predicted_price_today, actual_price=actual_price_today, accuracy=accuracy_today, predicted_prices_df=predicted_prices_df)

# Define daily update function to fetch the actual price daily and predict the next 7 days
def daily_update():
    available_stocks = list_available_stocks()
    for stock in available_stocks:
        model, scaler = load_model_and_scaler(stock)
        if model and scaler:
            stock_data = fetch_historical_data(stock)
            if stock_data is not None:
                # Fetch today's actual price and log it or save it somewhere
                actual_price_today = fetch_actual_price(stock)
                if actual_price_today is not None:
                    print(f"Actual price for {stock} on {pd.Timestamp.now().date()}: ₹{actual_price_today}")
                else:
                    print(f"Error fetching today's actual price for {stock}")
                
                # Perform 7-day predictions
                predict_next_7_days(stock, model, scaler, stock_data)

if __name__ == '__main__':
    # Schedule the daily update
    scheduler = BackgroundScheduler()
    scheduler.add_job(daily_update, 'interval', days=1)  # Run once every day
    scheduler.start()

    # Run the Flask app
    app.run(debug=True)
