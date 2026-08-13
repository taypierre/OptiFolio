import numpy as np
import pandas as pd
from data_ingestion import fetch_price_data, calculate_portfolio_stats

def calculate_bl_returns(mu_historical: pd.Series, sigma: pd.DataFrame, P: np.ndarray, Q: np.ndarray) -> pd.Series:
    """
    Calculates the Black-Litterman posterior expected returns.
    """
    # tau is a scalar indicating the uncertainty of the prior data
    tau = 0.05 
    
    # Omega is the uncertainty matrix of the views 
    # Use the diagonal of the covariance matrix scaled by tau to represent uncertainty in the views
    omega = np.dot(np.dot(P, tau * sigma), P.T)
    omega = np.diag(np.diag(omega)) # Only keep the diagonal elements to represent independent view uncertainties
    
    # Convert Pandas structures to Numpy arrays
    mu_hist_array = mu_historical.values.reshape(-1, 1) # Reshape to column vector
    sigma_array = sigma.values 
    Q = Q.reshape(-1, 1)
    
    # The Black-Litterman Master Equation (Bayesian updating)
    # This blends the historical returns with the views based on confidence levels
    part1 = np.linalg.inv(np.linalg.inv(tau * sigma_array) + np.dot(P.T, np.linalg.inv(omega)).dot(P))
    part2 = np.dot(np.linalg.inv(tau * sigma_array), mu_hist_array) + np.dot(P.T, np.linalg.inv(omega)).dot(Q)
    
    bl_mu = np.dot(part1, part2)
    
    return pd.Series(bl_mu.flatten(), index=mu_historical.index)

if __name__ == "__main__":
    tickers = ['AAPL', 'MSFT', 'JNJ', 'XOM']
    prices = fetch_price_data(tickers, '2020-01-01', '2023-01-01')
    mu, sigma = calculate_portfolio_stats(prices)
    
    # Define the Views using NumPy Arrays
    # View 1: AAPL returns 10%
    # View 2: MSFT outperforms XOM by 3%
    P = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, -1]
    ])
    
    Q = np.array([0.10, 0.03])
    
    print("\n--- Original Historical Expected Returns ---")
    print(mu.round(4))
    
    new_mu = calculate_bl_returns(mu, sigma, P, Q)
    
    print("\n--- New Black-Litterman Adjusted Returns ---")
    print(new_mu.round(4))