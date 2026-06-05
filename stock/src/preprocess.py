import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

class DataPreprocessor:
    def __init__(self, data):
        self.data = data
        self.scaler = MinMaxScaler()
        
    def handle_missing_values(self):
        """Handle missing values in the dataset"""
        print("Handling missing values...")
        print(f"Missing values before: {self.data.isnull().sum().sum()}")
        
        # Forward fill for small gaps
        self.data = self.data.fillna(method='ffill')
        # Backward fill for any remaining NaNs
        self.data = self.data.fillna(method='bfill')
        
        print(f"Missing values after: {self.data.isnull().sum().sum()}")
        return self
    
    def add_technical_indicators(self):
        """Add technical indicators for better analysis"""
        print("Adding technical indicators...")
        
        # Moving averages
        self.data['MA_7'] = self.data['Close'].rolling(window=7).mean()
        self.data['MA_20'] = self.data['Close'].rolling(window=20).mean()
        self.data['MA_50'] = self.data['Close'].rolling(window=50).mean()
        
        # RSI (Relative Strength Index)
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = self.data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = self.data['Close'].ewm(span=26, adjust=False).mean()
        self.data['MACD'] = exp1 - exp2
        self.data['Signal_Line'] = self.data['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        self.data['BB_Middle'] = self.data['Close'].rolling(window=20).mean()
        bb_std = self.data['Close'].rolling(window=20).std()
        self.data['BB_Upper'] = self.data['BB_Middle'] + (bb_std * 2)
        self.data['BB_Lower'] = self.data['BB_Middle'] - (bb_std * 2)
        
        # Volume indicators
        self.data['Volume_MA'] = self.data['Volume'].rolling(window=20).mean()
        self.data['Volume_Ratio'] = self.data['Volume'] / self.data['Volume_MA']
        
        # Price features
        self.data['Daily_Return'] = self.data['Close'].pct_change()
        self.data['Volatility'] = self.data['Daily_Return'].rolling(window=20).std()
        
        # Price range
        self.data['High_Low_Ratio'] = (self.data['High'] - self.data['Low']) / self.data['Close']
        self.data['Close_Open_Ratio'] = (self.data['Close'] - self.data['Open']) / self.data['Open']
        
        return self
    
    def create_lagged_features(self, lag_days=[1, 2, 3, 5]):
        """Create lagged features for time series prediction"""
        print("Creating lagged features...")
        
        for lag in lag_days:
            self.data[f'Close_Lag_{lag}'] = self.data['Close'].shift(lag)
            self.data[f'Volume_Lag_{lag}'] = self.data['Volume'].shift(lag)
            self.data[f'Return_Lag_{lag}'] = self.data['Daily_Return'].shift(lag)
        
        return self
    
    def create_target_variable(self, prediction_days=1):
        """Create target variable for prediction"""
        print(f"Creating target variable (next {prediction_days} day price)...")
        
        # Predict next day's price
        self.data['Target'] = self.data['Close'].shift(-prediction_days)
        
        # Binary classification target (price up or down)
        self.data['Target_Direction'] = (self.data['Target'] > self.data['Close']).astype(int)
        
        return self
    
    def remove_outliers(self, columns=None, threshold=3):
        """Remove outliers using Z-score method"""
        if columns is None:
            columns = ['Close', 'Volume', 'Daily_Return']
        
        print("Removing outliers...")
        initial_shape = self.data.shape
        
        for col in columns:
            if col in self.data.columns:
                z_scores = np.abs((self.data[col] - self.data[col].mean()) / self.data[col].std())
                self.data = self.data[z_scores < threshold]
        
        print(f"Removed {initial_shape[0] - self.data.shape[0]} rows with outliers")
        return self
    
    def scale_features(self, feature_columns=None):
        """Scale numerical features"""
        if feature_columns is None:
            feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 
                              'MA_7', 'MA_20', 'RSI', 'MACD', 'Volatility']
        
        # Select only columns that exist
        available_columns = [col for col in feature_columns if col in self.data.columns]
        
        print(f"Scaling features: {available_columns}")
        
        # Scale the features
        scaled_features = self.scaler.fit_transform(self.data[available_columns])
        
        # Create scaled dataframe
        scaled_df = pd.DataFrame(scaled_features, 
                                 columns=[f'{col}_scaled' for col in available_columns],
                                 index=self.data.index)
        
        # Combine with unscaled data
        self.data = pd.concat([self.data, scaled_df], axis=1)
        
        return self
    
    def clean_data(self):
        """Final cleaning step - remove NaN values"""
        print("Final data cleaning...")
        self.data = self.data.dropna()
        print(f"Final data shape: {self.data.shape}")
        return self
    
    def save_processed_data(self, ticker='AAPL'):
        """Save processed data to CSV"""
        os.makedirs('data/processed', exist_ok=True)
        filepath = f'data/processed/processed_{ticker}_stock_data.csv'
        self.data.to_csv(filepath)
        print(f"Processed data saved to {filepath}")
        return filepath

def load_processed_data(filepath):
    """Load processed data from CSV file"""
    data = pd.read_csv(filepath, index_col=0, parse_dates=True)
    print(f"Processed data loaded from {filepath}")
    return data

if __name__ == "__main__":
    # Test the preprocessor
    from src.data_loader import fetch_stock_data
    
    data = fetch_stock_data('AAPL', '2020-01-01', '2024-01-01')
    preprocessor = DataPreprocessor(data)
    
    processed_data = (preprocessor
                     .handle_missing_values()
                     .add_technical_indicators()
                     .create_lagged_features()
                     .create_target_variable()
                     .remove_outliers()
                     .scale_features()
                     .clean_data()
                     .data)
    
    print(processed_data.head())
    print(f"\nProcessed data columns: {processed_data.columns.tolist()}")
    preprocessor.save_processed_data()