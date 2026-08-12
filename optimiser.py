import numpy as np
import pandas as pd
import cvxpy as cp
from data_ingestion import fetch_price_data, calculate_portfolio_stats

def optimize_portfolio(mu: pd.Series, sigma: pd.DataFrame, risk_aversion: float = 2.0) -> pd.Series:
    """
    Solves for the optimal portfolio weights using Mean-Variance Optimization.
    """
    num_assets = len(mu)
    
    # 1. Define the optimization variable (the portfolio weights we want to find)
    weights = cp.Variable(num_assets)
    
    # Extract underlying numpy arrays to work smoothly with cvxpy
    mu_vals = mu.values
    sigma_vals = sigma.values
    
    # 2. Calculate expected portfolio return: w^T * mu
    expected_return = mu_vals @ weights
    
    # 3. Calculate portfolio variance: w^T * Sigma * w
    # cp.quad_form is a specialized cvxpy function required to keep the math convex
    portfolio_variance = cp.quad_form(weights, sigma_vals)
    
    # 4. Define the Objective Function: Maximize risk-adjusted return
    objective = cp.Maximize(expected_return - (risk_aversion / 2) * portfolio_variance)
    
    # 5. Define the Constraints
    constraints = [
        cp.sum(weights) == 1,  # All weights must sum to 100%
        weights >= 0           # No short selling (all weights must be positive)
    ]
    
    # 6. Solve the Quadratic Programming problem
    problem = cp.Problem(objective, constraints)
    problem.solve()
    
    # Return the optimized weights as a readable Pandas Series
    optimal_weights = pd.Series(weights.value, index=mu.index)
    
    # Clean up small floating point errors (e.g., turning 0.00000001 into 0)
    return optimal_weights.round(4)

if __name__ == "__main__":
    # Test the optimizer with our ingestion pipeline
    test_tickers = ['AAPL', 'MSFT', 'JNJ', 'XOM']
    prices = fetch_price_data(test_tickers, '2020-01-01', '2023-01-01')
    mu, sigma = calculate_portfolio_stats(prices)
    
    optimal_weights = optimize_portfolio(mu, sigma, risk_aversion=3.0)
    
    print("\n--- Optimal Portfolio Weights ---")
    # Convert weights to percentage format for readability
    print((optimal_weights * 100).astype(str) + '%')