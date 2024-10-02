import os
import pandas as pd
import numpy as np
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Bidirectional
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras_tuner.tuners import RandomSearch
from tensorflow.keras.optimizers import Adam

# Base directory to store saved models and scalers
MODEL_DIR = 'saved_models'

# Ensure the directory exists to store the models and scalers
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

class StockPricePredictor:
    def __init__(self, stock_name, window_size=120):
        self.stock_name = stock_name.upper()
        self.window_size = window_size
        self.model = None
        self.scaler = None
        self.X_train, self.X_val, self.y_train, self.y_val = None, None, None, None

    def fetch_historical_data(self):
        """Fetch historical stock data for the provided stock symbol."""
        try:
            stock_data = yf.download(self.stock_name + '.NS', period='5y', interval='1d')
            if stock_data.empty:
                raise ValueError(f"No data found for stock {self.stock_name}.")
            return stock_data
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    def prepare_data(self, stock_data):
        """Prepare data by calculating moving averages and scaling."""
        stock_data['MA10'] = stock_data['Close'].rolling(window=10).mean()
        stock_data['MA50'] = stock_data['Close'].rolling(window=50).mean()
        stock_data.fillna(method='backfill', inplace=True)

        features = stock_data[['Close', 'MA10', 'MA50']].dropna()
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_features = scaler.fit_transform(features)

        X, y = [], []
        for i in range(self.window_size, len(scaled_features)):
            X.append(scaled_features[i - self.window_size:i])
            y.append(scaled_features[i, 0])

        self.scaler = scaler
        return np.array(X), np.array(y)

    def build_cnn_lstm_model(self, hp):
        """Build a CNN-LSTM model with hyperparameter options."""
        model = Sequential()
        model.add(Conv1D(filters=hp.Int('cnn_filters', min_value=64, max_value=256, step=32),
                         kernel_size=hp.Choice('kernel_size', values=[3, 5]),
                         activation='relu',
                         input_shape=(self.window_size, 3)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)))
        model.add(Bidirectional(LSTM(units=hp.Int('lstm_units', min_value=64, max_value=256, step=32), return_sequences=False)))
        model.add(Dense(units=hp.Int('dense_units', min_value=32, max_value=128, step=32), activation='relu'))
        model.add(Dense(1))

        model.compile(optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
                      loss='mean_squared_error', metrics=['mae'])
        return model

    def split_data(self, X, y):
        """Split the data into training and validation sets."""
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning using Keras Tuner."""
        tuner = RandomSearch(self.build_cnn_lstm_model, objective='val_loss', max_trials=10, executions_per_trial=1,
                             directory=MODEL_DIR, project_name=f'{self.stock_name}_stock_prediction')

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        tuner.search(self.X_train, self.y_train, epochs=50, validation_data=(self.X_val, self.y_val), callbacks=[early_stopping])

        # Return the best model after the hyperparameter tuning
        self.model = tuner.get_best_models(num_models=1)[0]

    def train_model(self):
        """Train the best model for 100 epochs after hyperparameter tuning."""
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        self.model.fit(self.X_train, self.y_train, epochs=150, validation_data=(self.X_val, self.y_val), callbacks=[early_stopping])

    def save_model_and_scaler(self):
        """Save the trained model and scaler to disk."""
        model_path = os.path.join(MODEL_DIR, f'{self.stock_name}_best_model.h5')
        scaler_path = os.path.join(MODEL_DIR, f'{self.stock_name}_scaler.npy')
        self.model.save(model_path)
        np.save(scaler_path, self.scaler)
        print(f"Model and scaler saved to {MODEL_DIR} for stock {self.stock_name}.")

    def execute_pipeline(self):
        """Execute the complete pipeline: data fetching, preparation, tuning, training, and saving."""
        stock_data = self.fetch_historical_data()
        if stock_data is None:
            print(f"Failed to fetch data for {self.stock_name}.")
            return

        X, y = self.prepare_data(stock_data)
        self.split_data(X, y)
        print(f"Starting hyperparameter tuning for {self.stock_name}...")
        self.hyperparameter_tuning()
        print(f"Hyperparameter tuning completed. Training best model for {self.stock_name}...")
        self.train_model()
        self.save_model_and_scaler()


if __name__ == '__main__':
    stock_name = input("Enter the NSE stock symbol (e.g., RELIANCE, TCS, HDFCBANK): ").upper()
    
    predictor = StockPricePredictor(stock_name)
    predictor.execute_pipeline()
