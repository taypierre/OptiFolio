import yfinance as yf
import pandas as pd
import numpy as np

def fetch_price_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches historical prices for a list of tickers.
    """
    # Fetch the raw multi-index dataframe
    raw_data = yf.download(tickers, start=start_date, end=end_date)
    
    if 'Adj Close' in raw_data:
        prices = raw_data['Adj Close']
    elif 'Close' in raw_data:
        prices = raw_data['Close']
    else:
        prices = raw_data.xs('Close', level=1, axis=1)
        
    return prices.dropna()

def calculate_portfolio_stats(prices: pd.DataFrame, trading_days: int = 252) -> tuple:
    """
    Calculates annualized returns and the covariance matrix from price data.
    """
    # Calculate daily logarithmic returns
    daily_returns = np.log(prices / prices.shift(1)).dropna()
    
    # Calculate annualized expected returns (vector mu)
    annualized_returns = daily_returns.mean() * trading_days
    
    # Calculate annualized covariance matrix (matrix Sigma)
    covariance_matrix = daily_returns.cov() * trading_days
    
    return annualized_returns, covariance_matrix

# --- Test ---
if __name__ == "__main__":
    test_tickers = ['AAPL', 'MSFT', 'JNJ', 'XOM']
    
    prices = fetch_price_data(test_tickers, '2020-01-01', '2023-01-01')
    mu, sigma = calculate_portfolio_stats(prices)
    
    print("\n--- Annualized Expected Returns (\u03bc) ---")
    print(mu)
    print("\n--- Annualized Covariance Matrix (\u03a3) ---")
    print(sigma)