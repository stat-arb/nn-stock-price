import requests
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn.metrics import mean_squared_error
import warnings
from newsapi import NewsApiClient
from transformers import pipeline

warnings.filterwarnings("ignore")

# Constants
days_to_predict = 30
stock_ticker = 'spy'
look_back = 90

# Initialize NewsAPI client
newsapi = NewsApiClient(api_key='YOUR_NEWS_API_KEY')

# Initialize sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Function to fetch stock data
def fetch_stock_data(ticker):
    url = f'http://localhost:8000/stock/{ticker}'
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        data = response.json()
        return pd.DataFrame(data)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

# Function to add features like moving averages
def add_features(data, ticker):
    data['SMA_50'] = data['Close'].rolling(window=50).mean()  # 50-day Simple Moving Average
    data['SMA_200'] = data['Close'].rolling(window=200).mean()  # 200-day Simple Moving Average
    data['Price_Change'] = data['Close'].pct_change()  # Price percentage change
    data.fillna(method='bfill', inplace=True)  # Handle missing data by backfilling
    
    # Add sentiment analysis
    news = fetch_news(ticker)
    data['Sentiment'] = analyze_sentiment(news)
    
    return data

# Function to prepare data for LSTM model
def prepare_data(df, look_back):
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df[['Close']])

    X, y = [], []
    for i in range(look_back, len(df_scaled)):
        X.append(df_scaled[i-look_back:i, 0])
        y.append(df_scaled[i, 0])
    X, y = np.array(X), np.array(y)
    
    # Reshape input data for LSTM (samples, time steps, features)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler

# Function to build the LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(Bidirectional(LSTM(100, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(100, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(100)))
    model.add(Dropout(0.3))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to predict the next 'days_to_predict' days
def predict_future(model, last_days, days_to_predict, scaler):
    predicted_prices_scaled = []
    
    for _ in range(days_to_predict):
        prediction = model.predict(last_days)
        predicted_prices_scaled.append(prediction[0, 0])
        
        # Update the last_days array for next prediction
        last_days = np.roll(last_days, shift=-1)
        last_days[-1] = prediction
        last_days = np.array([last_days])
    
    # Inverse transform to get actual prices
    predicted_prices = scaler.inverse_transform(np.array(predicted_prices_scaled).reshape(-1, 1))
    return predicted_prices

def fetch_news(ticker):
    try:
        news = newsapi.get_everything(q=ticker, language='en', sort_by='publishedAt', page_size=10)
        return news['articles']
    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")
        return []

def analyze_sentiment(news):
    sentiments = []
    for article in news:
        result = sentiment_pipeline(article['title'])[0]
        sentiments.append(result['score'] if result['label'] == 'POSITIVE' else -result['score'])
    return np.mean(sentiments) if sentiments else 0

# Main function to execute the workflow
def main():
    data = fetch_stock_data(stock_ticker)
    if data is None:
        return  # Exit if fetching data failed

    data = add_features(data, stock_ticker)  # Add additional features
    print(f"Stock data for {stock_ticker}:")
    print(data.tail())

    # Prepare data for LSTM model
    X, y, scaler = prepare_data(data, look_back)
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Try to load existing model, if not found, create and train a new one
    try:
        model = load_model(f'{stock_ticker}_model.h5')
        print("Loaded existing model.")
    except:
        print("Training new model.")
        model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)
        model.save(f'{stock_ticker}_model.h5')

    # Make predictions on the test set
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Evaluate the model
    mse = mean_squared_error(y_test_actual, predicted_prices)
    rmse = np.sqrt(mse)
    print(f"RMSE on Test Data: {rmse}")

    # Predict future prices
    last_days = X_test[-1]  # Use the last test data as a base for future predictions
    predicted_future_prices = predict_future(model, last_days, days_to_predict, scaler)

    print(f"\nPredicted Prices for the Next {days_to_predict} Days:")
    print(predicted_future_prices)

if __name__ == "__main__":
    main()
