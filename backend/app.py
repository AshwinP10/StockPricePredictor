from flask import Flask, request, render_template
import yfinance as yf
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import io
import base64
import os

app = Flask(__name__)

# Path to the model file
model_path = 'model.pkl'

def load_model():
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
    else:
        print(f"Model file not found. Creating a new model.")
        model = RandomForestRegressor(n_estimators=100)
    return model

def fetch_stock_data(ticker):
    print(f"Fetching data for {ticker}")
    stock = yf.Ticker(ticker)
    data = stock.history(period='1y')
    data.reset_index(inplace=True)
    return data

def preprocess_data(data):
    print("Preprocessing data")
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    data['Open'] = data['Open'].pct_change().fillna(0)
    data['High'] = data['High'].pct_change().fillna(0)
    data['Low'] = data['Low'].pct_change().fillna(0)
    data['Close'] = data['Close'].pct_change().fillna(0)
    features = data[['Open', 'High', 'Low']].values
    target = data['Close'].shift(-1).dropna()
    features = features[:len(target)]
    return features, target

def train_model(ticker):
    print(f"Training model for {ticker}")
    data = fetch_stock_data(ticker)
    features, target = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    print(f"Saving model to {model_path}")
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)

def make_prediction(ticker):
    model = load_model()
    print(f"Making prediction for {ticker}")
    data = fetch_stock_data(ticker)
    features, _ = preprocess_data(data)
    if len(features) == 0:
        return None
    return model.predict([features[-1]])[0]

def plot_stock_data(ticker):
    print(f"Plotting stock data for {ticker}")
    data = fetch_stock_data(ticker)
    fig, ax = plt.subplots()
    ax.plot(data.index, data['Close'], label='Close Price')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title(f'Stock Prices for {ticker}')
    ax.legend()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close(fig)
    return img_base64

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form['ticker'].upper()
        train_model(ticker)  # Train and save the model for the new ticker
        prediction = make_prediction(ticker)  # Load and use the trained model for prediction
        stock_image = plot_stock_data(ticker)
        return render_template('results.html', ticker=ticker, prediction=prediction, stock_image=stock_image)
    return '''
        <form method="post">
            <input type="text" name="ticker" placeholder="Enter stock ticker" required>
            <input type="submit" value="Get Prediction">
        </form>
    '''

if __name__ == '__main__':
    app.run(debug=True)

