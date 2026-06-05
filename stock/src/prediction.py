import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

class StockPredictor:
    def __init__(self, data, target_col='Target'):
        self.data = data
        self.target_col = target_col
        self.models = {}
        self.results = {}
        
    def prepare_features(self, feature_cols=None):
        """Prepare features for modeling"""
        if feature_cols is None:
            # Default features
            feature_cols = ['Close', 'Volume', 'MA_7', 'MA_20', 'RSI', 
                           'MACD', 'Volatility', 'Daily_Return']
        
        # Select only available columns
        available_features = [col for col in feature_cols if col in self.data.columns]
        
        # Add lagged features if available
        lag_cols = [col for col in self.data.columns if 'Lag' in col]
        available_features.extend(lag_cols[:10])  # Limit to 10 lag features
        
        # Add scaled features if available
        scaled_cols = [col for col in self.data.columns if 'scaled' in col]
        available_features.extend(scaled_cols[:10])
        
        print(f"Using {len(available_features)} features: {available_features[:10]}...")
        
        X = self.data[available_features]
        y = self.data[self.target_col]
        
        # Remove any remaining NaN values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        return X, y
    
    def train_test_split_time_series(self, X, y, test_size=0.2):
        """Split data chronologically for time series"""
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def train_models(self, X_train, y_train):
        """Train multiple models for comparison"""
        print("\nTraining models...")
        
        # Random Forest
        print("Training Random Forest...")
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        self.models['Random Forest'] = rf
        
        # Gradient Boosting
        print("Training Gradient Boosting...")
        gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb.fit(X_train, y_train)
        self.models['Gradient Boosting'] = gb
        
        # Linear Regression
        print("Training Linear Regression...")
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        self.models['Linear Regression'] = lr
        
        print("All models trained successfully!")
        
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        print("\nEvaluating models...")
        
        for name, model in self.models.items():
            # Make predictions
            predictions = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, predictions)
            
            # Calculate MAPE
            mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
            
            # Store results
            self.results[name] = {
                'MSE': mse,
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'MAPE': mape,
                'Predictions': predictions
            }
            
            print(f"\n{name}:")
            print(f"  RMSE: ${rmse:.2f}")
            print(f"  MAE: ${mae:.2f}")
            print(f"  R² Score: {r2:.4f}")
            print(f"  MAPE: {mape:.2f}%")
        
        return self.results
    
    def predict_future(self, days=30):
        """Predict future prices"""
        print(f"\nPredicting next {days} days...")
        
        # Use the best model (Random Forest)
        best_model = self.models['Random Forest']
        
        # Get the last available features
        X_full, _ = self.prepare_features()
        last_features = X_full.iloc[-1:].values
        
        predictions = []
        current_features = last_features.copy()
        
        for i in range(days):
            # Make prediction
            next_price = best_model.predict(current_features)[0]
            predictions.append(next_price)
            
            # Update features (simplified - in reality would need to update all features)
            # For demonstration, we'll just shift and add noise
            current_features = current_features * (1 + np.random.normal(0, 0.01, current_features.shape))
        
        return np.array(predictions)
    
    def feature_importance(self):
        """Get feature importance from Random Forest model"""
        if 'Random Forest' in self.models:
            rf_model = self.models['Random Forest']
            
            # Get feature names
            X, _ = self.prepare_features()
            feature_names = X.columns
            
            # Get importance scores
            importance_scores = rf_model.feature_importances_
            
            # Create sorted dataframe
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance_scores
            }).sort_values('Importance', ascending=False)
            
            print("\nTop 10 Most Important Features:")
            print(importance_df.head(10))
            
            return importance_df
        
        return None
    
    def run_prediction_pipeline(self):
        """Run complete prediction pipeline"""
        print("="*50)
        print("STOCK PRICE PREDICTION PIPELINE")
        print("="*50)
        
        # Prepare features
        X, y = self.prepare_features()
        
        # Split data
        X_train, X_test, y_train, y_test = self.train_test_split_time_series(X, y)
        
        # Train models
        self.train_models(X_train, y_train)
        
        # Evaluate models
        self.evaluate_models(X_test, y_test)
        
        # Feature importance
        importance_df = self.feature_importance()
        
        # Future predictions
        future_prices = self.predict_future(30)
        
        return {
            'results': self.results,
            'feature_importance': importance_df,
            'future_predictions': future_prices
        }

if __name__ == "__main__":
    from preprocess import load_processed_data
    
    # Load processed data
    data = load_processed_data('data/processed/processed_AAPL_stock_data.csv')
    
    # Initialize predictor
    predictor = StockPredictor(data)
    
    # Run prediction pipeline
    results = predictor.run_prediction_pipeline()