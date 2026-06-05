import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class StockAnalyzer:
    def __init__(self, data):
        self.data = data
        self.results = {}
        
    def calculate_returns_statistics(self):
        """Calculate comprehensive return statistics"""
        print("Calculating return statistics...")
        
        if 'Daily_Return' not in self.data.columns:
            self.data['Daily_Return'] = self.data['Close'].pct_change()
        
        returns = self.data['Daily_Return'].dropna()
        
        stats_dict = {
            'Mean Daily Return': returns.mean(),
            'Median Daily Return': returns.median(),
            'Std Dev Daily Return': returns.std(),
            'Skewness': returns.skew(),
            'Kurtosis': returns.kurtosis(),
            'Sharpe Ratio (Annualized)': (returns.mean() / returns.std()) * np.sqrt(252),
            'Maximum Drawdown': self.calculate_max_drawdown(),
            'Value at Risk (95%)': returns.quantile(0.05),
            'Conditional VaR (95%)': returns[returns <= returns.quantile(0.05)].mean(),
            'Positive Days %': (returns > 0).mean() * 100,
            'Negative Days %': (returns < 0).mean() * 100
        }
        
        self.results['Returns Statistics'] = stats_dict
        return stats_dict
    
    def calculate_max_drawdown(self):
        """Calculate maximum drawdown"""
        cumulative_returns = (1 + self.data['Daily_Return']).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown.min()
    
    def analyze_trends(self):
        """Analyze price trends and patterns"""
        print("Analyzing price trends...")
        
        trends = {}
        
        # Overall trend
        start_price = self.data['Close'].iloc[0]
        end_price = self.data['Close'].iloc[-1]
        trends['Total Return'] = ((end_price - start_price) / start_price) * 100
        trends['CAGR'] = ((end_price / start_price) ** (252/len(self.data))) - 1
        
        # Moving average crossovers
        if 'MA_7' in self.data.columns and 'MA_20' in self.data.columns:
            ma_7 = self.data['MA_7'].iloc[-1]
            ma_20 = self.data['MA_20'].iloc[-1]
            trends['Current Trend'] = 'Bullish' if ma_7 > ma_20 else 'Bearish'
            
            # Count crossovers
            crossovers = ((self.data['MA_7'] > self.data['MA_20']) != 
                         (self.data['MA_7'].shift(1) > self.data['MA_20'].shift(1)))
            trends['Number of Crossovers'] = crossovers.sum()
        
        # Volatility analysis
        if 'Volatility' in self.data.columns:
            trends['Current Volatility'] = self.data['Volatility'].iloc[-1]
            trends['Avg Volatility'] = self.data['Volatility'].mean()
            trends['Volatility Trend'] = 'Increasing' if self.data['Volatility'].iloc[-1] > self.data['Volatility'].mean() else 'Decreasing'
        
        self.results['Trend Analysis'] = trends
        return trends
    
    def analyze_volume_patterns(self):
        """Analyze volume patterns"""
        print("Analyzing volume patterns...")
        
        volume_stats = {}
        
        if 'Volume' in self.data.columns:
            volume_stats['Avg Daily Volume'] = self.data['Volume'].mean()
            volume_stats['Max Volume Day'] = self.data['Volume'].idxmax()
            volume_stats['Volume on Up Days'] = self.data[self.data['Daily_Return'] > 0]['Volume'].mean()
            volume_stats['Volume on Down Days'] = self.data[self.data['Daily_Return'] < 0]['Volume'].mean()
            volume_stats['Volume Price Correlation'] = self.data['Volume'].corr(self.data['Daily_Return'])
            
            # Price-volume relationship
            price_up_volume_up = ((self.data['Daily_Return'] > 0) & 
                                 (self.data['Volume'] > self.data['Volume'].shift(1))).sum()
            volume_stats['Strong Up Days'] = price_up_volume_up
        
        self.results['Volume Analysis'] = volume_stats
        return volume_stats
    
    def seasonal_analysis(self):
        """Analyze seasonal patterns"""
        print("Analyzing seasonal patterns...")
        
        seasonal = {}
        
        # Monthly returns
        monthly_returns = self.data.groupby('Month')['Daily_Return'].mean() * 100
        seasonal['Best Month'] = monthly_returns.idxmax(), monthly_returns.max()
        seasonal['Worst Month'] = monthly_returns.idxmin(), monthly_returns.min()
        seasonal['Monthly Returns'] = monthly_returns.to_dict()
        
        # Day of week effects
        if 'DayOfWeek' in self.data.columns:
            daily_returns = self.data.groupby('DayOfWeek')['Daily_Return'].mean() * 100
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
            seasonal['Daily Patterns'] = {day_names[i]: daily_returns[i] for i in range(len(daily_returns))}
        
        self.results['Seasonal Analysis'] = seasonal
        return seasonal
    
    def correlation_analysis(self):
        """Analyze correlations between features"""
        print("Analyzing correlations...")
        
        # Select numeric columns for correlation
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.data[numeric_cols].corr()
        
        # Find top correlations with Close price
        close_correlations = correlation_matrix['Close'].sort_values(ascending=False)
        
        correlations = {
            'Top Positive Correlations': close_correlations.head(10).to_dict(),
            'Top Negative Correlations': close_correlations.tail(10).to_dict()
        }
        
        self.results['Correlation Analysis'] = correlations
        return correlations
    
    def risk_analysis(self):
        """Perform comprehensive risk analysis"""
        print("Performing risk analysis...")
        
        risk_metrics = {}
        
        if 'Daily_Return' in self.data.columns:
            returns = self.data['Daily_Return'].dropna()
            
            # Beta calculation (using S&P500 as market proxy)
            # Simplified: using index returns as proxy
            risk_metrics['Beta'] = 1.0  # In real scenario, calculate against market index
            
            # Downside risk
            downside_returns = returns[returns < 0]
            risk_metrics['Downside Deviation'] = downside_returns.std()
            risk_metrics['Sortino Ratio'] = (returns.mean() / downside_returns.std()) * np.sqrt(252)
            
            # Maximum drawdown details
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            risk_metrics['Max Drawdown'] = drawdown.min()
            risk_metrics['Drawdown Duration'] = (drawdown < 0).sum()
            
            # VaR for different confidence levels
            risk_metrics['VaR 90%'] = returns.quantile(0.10)
            risk_metrics['VaR 95%'] = returns.quantile(0.05)
            risk_metrics['VaR 99%'] = returns.quantile(0.01)
        
        self.results['Risk Analysis'] = risk_metrics
        return risk_metrics
    
    def generate_full_report(self):
        """Generate complete analysis report"""
        print("\n" + "="*50)
        print("STOCK ANALYSIS REPORT")
        print("="*50)
        
        # Run all analyses
        self.calculate_returns_statistics()
        self.analyze_trends()
        self.analyze_volume_patterns()
        self.seasonal_analysis()
        self.correlation_analysis()
        self.risk_analysis()
        
        # Print summary
        for category, metrics in self.results.items():
            print(f"\n{category}:")
            print("-" * 30)
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"{key:25}: {value:.4f}")
                else:
                    print(f"{key:25}: {value}")
        
        return self.results

if __name__ == "__main__":
    from src.preprocess import load_processed_data
    
    # Load processed data
    data = load_processed_data('data/processed/processed_AAPL_stock_data.csv')
    
    # Analyze
    analyzer = StockAnalyzer(data)
    report = analyzer.generate_full_report()