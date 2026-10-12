import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

def fetch_stock_data(ticker='AAPL', start_date='2020-01-01', end_date='2024-01-01'):
    """
    Fetch stock data from Yahoo Finance
    
    Parameters:
    ticker (str): Stock ticker symbol
    start_date (str): Start date in YYYY-MM-DD format
    end_date (str): End date in YYYY-MM-DD format
    
    Returns:
    pd.DataFrame: Stock data
    """
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    
    # Download data
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    
    # Add additional features
    stock_data['Ticker'] = ticker
    stock_data['Day'] = stock_data.index.day
    stock_data['Month'] = stock_data.index.month
    stock_data['Year'] = stock_data.index.year
    stock_data['DayOfWeek'] = stock_data.index.dayofweek
    
    return stock_data

def save_raw_data(data, ticker='AAPL'):
    """
    Save raw data to CSV file
    """
    os.makedirs('data/raw', exist_ok=True)
    filepath = f'data/raw/{ticker}_stock_data.csv'
    data.to_csv(filepath)
    print(f"Data saved to {filepath}")
    return filepath

def load_raw_data(filepath):
    """
    Load raw data from CSV file
    """
    data = pd.read_csv(filepath, index_col=0, parse_dates=True)
    print(f"Data loaded from {filepath}")
    return data

if __name__ == "__main__":
    # Test the data loader
    data = fetch_stock_data('AAPL', '2020-01-01', '2024-01-01')
    print(data.head())
    print(f"\nData shape: {data.shape}")
    save_raw_data(data)