from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Optional
import os
import json
import yfinance as yf
from openai import OpenAI
import logging
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import talib

app = FastAPI()
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load OpenAI API key from environment variable or config file
openai_api_key = os.getenv("OPENAI_API_KEY")
config_file_path = '/etc/python-gpt.json'

if not openai_api_key:
    if os.path.exists(config_file_path):
        with open(config_file_path) as config_file:
            config = json.load(config_file)
            openai_api_key = config.get('OPENAI_API_KEY')
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in environment or config file")

# Initialize the OpenAI client
client = OpenAI(api_key=openai_api_key)

# Utility function to fetch stock data using yfinance
def fetch_stock_data(ticker: str, period: str = "120d"):
    try:
        stock_data = yf.download(ticker, period=period)
        if stock_data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for ticker {ticker}")
        return stock_data.reset_index().to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching stock data: {str(e)}")

# API Endpoint to retrieve stock data
@app.get("/stock/{ticker}")
@limiter.limit("5/minute")
def get_stock_data(ticker: str, period: Optional[str] = "120d", request=Depends(get_remote_address)):
    """
    Get stock data for a given ticker.
    :param ticker: Stock ticker symbol (e.g. 'AAPL').
    :param period: Optional period (default is 120 days). Examples: "1d", "5d", "1mo", "1y".
    """
    try:
        return fetch_stock_data(ticker, period)
    except Exception as e:
        logger.error(f"Error in get_stock_data: {str(e)}")
        raise

# API Endpoint to chat with OpenAI GPT model
@app.post("/chat")
@limiter.limit("10/minute")
def chat(data: Dict[str, str], request=Depends(get_remote_address)):
    """
    Interact with the GPT-3.5 model by sending a message.
    :param data: A dictionary with a "message" key.
    """
    message = data.get("message")
    if not message:
        raise HTTPException(status_code=400, detail="Invalid request: 'message' is required")

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": message}
            ]
        )
        return {"message": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating chat response: {str(e)}")

# API Endpoint to retrieve technical indicators
@app.get("/technical_indicators/{ticker}")
@limiter.limit("5/minute")
def get_technical_indicators(ticker: str, period: Optional[str] = "120d", request=Depends(get_remote_address)):
    """
    Get technical indicators for a given stock ticker.
    """
    try:
        stock_data = fetch_stock_data(ticker, period)
        df = pd.DataFrame(stock_data)
        
        # Calculate technical indicators
        df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)
        df['SMA_200'] = talib.SMA(df['Close'], timeperiod=200)
        df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
        df['MACD'], df['MACD_Signal'], _ = talib.MACD(df['Close'])
        
        return df.to_dict(orient='records')
    except Exception as e:
        logger.error(f"Error in get_technical_indicators: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error calculating technical indicators: {str(e)}")

