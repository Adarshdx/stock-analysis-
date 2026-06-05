import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

class StockVisualizer:
    def __init__(self, data):
        self.data = data
        self.set_style()
        
    def set_style(self):
        """Set plotting style"""
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
    def create_price_chart(self, save_path='reports/figures/price_chart.png'):
        """Create comprehensive price chart"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
        
        # Price with moving averages
        axes[0].plot(self.data.index, self.data['Close'], label='Close Price', linewidth=2)
        if 'MA_20' in self.data.columns:
            axes[0].plot(self.data.index, self.data['MA_20'], label='20-day MA', alpha=0.7)
        if 'MA_50' in self.data.columns:
            axes[0].plot(self.data.index, self.data['MA_50'], label='50-day MA', alpha=0.7)
        axes[0].set_title('Stock Price with Moving Averages', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Price ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Volume
        colors = ['red' if self.data['Close'].iloc[i] < self.data['Open'].iloc[i] 
                  else 'green' for i in range(len(self.data))]
        axes[1].bar(self.data.index, self.data['Volume'], color=colors, alpha=0.7)
        axes[1].set_title('Trading Volume', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Volume')
        axes[1].grid(True, alpha=0.3)
        
        # RSI
        if 'RSI' in self.data.columns:
            axes[2].plot(self.data.index, self.data['RSI'], label='RSI', color='purple')
            axes[2].axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought (70)')
            axes[2].axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold (30)')
            axes[2].fill_between(self.data.index, 30, 70, alpha=0.1, color='gray')
            axes[2].set_title('Relative Strength Index (RSI)', fontsize=14, fontweight='bold')
            axes[2].set_ylabel('RSI')
            axes[2].set_ylim(0, 100)
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        plt.xlabel('Date')
        plt.tight_layout()
        
        os.makedirs('reports/figures', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Chart saved to {save_path}")
        
    def create_returns_distribution(self, save_path='reports/figures/returns_distribution.png'):
        """Create returns distribution plot"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Histogram of returns
        returns = self.data['Daily_Return'].dropna()
        axes[0].hist(returns, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0].axvline(x=returns.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {returns.mean():.4f}')
        axes[0].axvline(x=returns.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {returns.median():.4f}')
        axes[0].set_title('Distribution of Daily Returns', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Daily Return')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Q-Q plot for normality check
        from scipy import stats
        stats.probplot(returns, dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot (Normality Check)', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Chart saved to {save_path}")
        
    def create_correlation_heatmap(self, save_path='reports/figures/correlation_heatmap.png'):
        """Create correlation heatmap"""
        # Select numeric columns for correlation
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return']
        available_cols = [col for col in numeric_cols if col in self.data.columns]
        
        # Add technical indicators if available
        indicators = ['MA_7', 'MA_20', 'RSI', 'MACD', 'Volatility']
        available_cols.extend([col for col in indicators if col in self.data.columns])
        
        corr_matrix = self.data[available_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Chart saved to {save_path}")
        
    def create_candlestick_chart(self, days=60, save_path='reports/figures/candlestick_chart.html'):
        """Create interactive candlestick chart using Plotly"""
        # Get last N days
        plot_data = self.data.tail(days).copy()
        plot_data = plot_data.reset_index()
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.03, 
                           subplot_titles=(f'Candlestick Chart (Last {days} days)', 'Volume'),
                           row_width=[0.2, 0.7])
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(x=plot_data['Date'],
                                    open=plot_data['Open'],
                                    high=plot_data['High'],
                                    low=plot_data['Low'],
                                    close=plot_data['Close'],
                                    name='Price'),
                     row=1, col=1)
        
        # Add moving averages
        if 'MA_20' in plot_data.columns:
            fig.add_trace(go.Scatter(x=plot_data['Date'], 
                                    y=plot_data['MA_20'],
                                    name='20-day MA',
                                    line=dict(color='orange', width=1)),
                         row=1, col=1)
        
        # Volume bars
        colors = ['red' if plot_data['Close'].iloc[i] < plot_data['Open'].iloc[i] 
                  else 'green' for i in range(len(plot_data))]
        
        fig.add_trace(go.Bar(x=plot_data['Date'],
                            y=plot_data['Volume'],
                            name='Volume',
                            marker_color=colors),
                     row=2, col=1)
        
        # Update layout
        fig.update_layout(title='Interactive Stock Analysis',
                         xaxis_title='Date',
                         yaxis_title='Price ($)',
                         template='plotly_dark',
                         height=800)
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        os.makedirs('reports/figures', exist_ok=True)
        fig.write_html(save_path)
        print(f"Interactive chart saved to {save_path}")
        return fig
        
    def create_prediction_plot(self, predictions, actual=None, save_path='reports/figures/predictions.png'):
        """Create prediction visualization"""
        plt.figure(figsize=(12, 6))
        
        # Plot predictions
        plt.plot(range(len(predictions)), predictions, label='Predictions', 
                marker='o', linewidth=2, markersize=4)
        
        # Add confidence interval (simulated)
        std_dev = np.std(predictions) * 0.1
        upper_bound = predictions + (1.96 * std_dev)
        lower_bound = predictions - (1.96 * std_dev)
        
        plt.fill_between(range(len(predictions)), lower_bound, upper_bound, 
                        alpha=0.2, color='blue', label='95% Confidence Interval')
        
        plt.title('Stock Price Predictions (Next 30 Days)', fontsize=14, fontweight='bold')
        plt.xlabel('Days Ahead', fontsize=12)
        plt.ylabel('Predicted Price ($)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Prediction plot saved to {save_path}")
        
    def create_monthly_returns_heatmap(self, save_path='reports/figures/monthly_returns.png'):
        """Create monthly returns heatmap"""
        if 'Daily_Return' not in self.data.columns or 'Year' not in self.data.columns:
            print("Required columns not found for monthly returns heatmap")
            return
        
        # Calculate monthly returns
        monthly_returns = self.data.groupby(['Year', 'Month'])['Daily_Return'].apply(
            lambda x: (1 + x).prod() - 1
        ).unstack()
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(monthly_returns, annot=True, fmt='.2%', cmap='RdYlGn', 
                   center=0, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Monthly Returns Heatmap', fontsize=14, fontweight='bold')
        plt.xlabel('Month')
        plt.ylabel('Year')
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Monthly returns heatmap saved to {save_path}")
        
    def create_rolling_statistics(self, window=20, save_path='reports/figures/rolling_stats.png'):
        """Create rolling statistics plot"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Rolling mean and std
        rolling_mean = self.data['Close'].rolling(window=window).mean()
        rolling_std = self.data['Close'].rolling(window=window).std()
        
        axes[0,0].plot(self.data.index, self.data['Close'], label='Actual', alpha=0.5)
        axes[0,0].plot(self.data.index, rolling_mean, label=f'{window}-day MA', color='red')
        axes[0,0].fill_between(self.data.index, rolling_mean - rolling_std, 
                               rolling_mean + rolling_std, alpha=0.2, color='red')
        axes[0,0].set_title(f'Rolling Statistics (Window={window})', fontsize=12, fontweight='bold')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Rolling volatility
        if 'Volatility' in self.data.columns:
            axes[0,1].plot(self.data.index, self.data['Volatility'], color='orange')
            axes[0,1].set_title('Rolling Volatility (20-day)', fontsize=12, fontweight='bold')
            axes[0,1].set_ylabel('Volatility')
            axes[0,1].grid(True, alpha=0.3)
        
        # Rolling Sharpe ratio
        rolling_returns = self.data['Daily_Return'].rolling(window=window)
        rolling_sharpe = (rolling_returns.mean() / rolling_returns.std()) * np.sqrt(252)
        axes[1,0].plot(self.data.index, rolling_sharpe, color='green')
        axes[1,0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1,0].set_title('Rolling Sharpe Ratio', fontsize=12, fontweight='bold')
        axes[1,0].set_ylabel('Sharpe Ratio')
        axes[1,0].grid(True, alpha=0.3)
        
        # Rolling correlation between price and volume
        if 'Volume' in self.data.columns:
            rolling_corr = self.data['Close'].rolling(window=window).corr(self.data['Volume'])
            axes[1,1].plot(self.data.index, rolling_corr, color='purple')
            axes[1,1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[1,1].set_title('Price-Volume Rolling Correlation', fontsize=12, fontweight='bold')
            axes[1,1].set_ylabel('Correlation')
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Rolling statistics plot saved to {save_path}")
        
    def create_dashboard(self, predictions=None):
        """Create comprehensive dashboard"""
        print("Creating comprehensive dashboard...")
        
        # Create all visualizations
        self.create_price_chart()
        self.create_returns_distribution()
        self.create_correlation_heatmap()
        self.create_monthly_returns_heatmap()
        self.create_rolling_statistics()
        
        if predictions is not None:
            self.create_prediction_plot(predictions)
        
        # Create interactive candlestick chart
        self.create_candlestick_chart()
        
        print("\nDashboard created successfully!")

if __name__ == "__main__":
    from preprocess import load_processed_data
    
    # Load data
    data = load_processed_data('data/processed/processed_AAPL_stock_data.csv')
    
    # Create visualizations
    visualizer = StockVisualizer(data)
    visualizer.create_dashboard()