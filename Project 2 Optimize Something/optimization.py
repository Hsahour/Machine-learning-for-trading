"""
MC1-P2: Optimize a portfolio.

Copyright 2018, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 4646/7646

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as github and gitlab.  This copyright statement should not be removed
or edited.

We do grant permission to share solutions privately with non-students such
as potential employers. However, sharing with other current or future
students of CS 7646 is prohibited and subject to being investigated as a
GT honor code violation.

-----do not edit anything above this line---

Student Name: Hossein Sahour                                        
GT User ID: hsahour3                                        
GT ID: 903941641  
"""


import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as spo
from util import get_data
import os

def compute_portfolio_stats(prices, allocs, rfr=0.0, sf=252.0):
    """
    Compute portfolio statistics.
    """
    # Normalized prices
    norm_price = prices / prices.iloc[0]
    # Allocated portfolio
    alloced = norm_price * allocs
    # Portfolio values
    pos_vals = alloced
    # Portfolio value
    port_val = pos_vals.sum(axis=1)

    # Daily returns
    daily_returns = port_val.pct_change(1).dropna()

    # Cumulative return
    cr = (port_val[-1] / port_val[0]) - 1

    # Average daily return
    adr = daily_returns.mean()

    # Standard deviation of daily return
    sddr = daily_returns.std()

    # Sharpe Ratio
    sr = (adr - rfr) / sddr * np.sqrt(sf)

    return cr, adr, sddr, sr

def optimize_allocations(prices):
    """
    Optimize portfolio allocations for maximum Sharpe Ratio.
    """
    def sharpe_ratio(allocs):
        _, _, _, sr = compute_portfolio_stats(prices, allocs)
        return -sr  # Negative for minimization

    num_assets = len(prices.columns)
    initial_guess = np.ones(num_assets) / num_assets
    bounds = tuple((0, 1) for _ in range(num_assets))
    constraints = {'type': 'eq', 'fun': lambda inputs: 1 - np.sum(inputs)}

    result = spo.minimize(sharpe_ratio, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    if result.success:
        return result.x
    else:
        raise Exception("Optimization did not converge")

def plot_data(portfolio, spy, title="Daily Portfolio vs SPY"):
    plt.figure(figsize=(10, 6))
    plt.plot(portfolio, label="Optimized Portfolio")
    plt.plot(spy, label="SPY")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Normalized Price")
    plt.legend()
    plt.grid()
    plt.savefig('Figure1.png')

def optimize_portfolio(sd=dt.datetime(2008, 6, 1), ed=dt.datetime(2009, 6, 1), syms=["IBM", "X", "GLD", "JPM"], gen_plot=False):
    """
    Optimize the portfolio and generate plot
    """
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)
    prices = prices_all[syms]
    prices_SPY = prices_all['SPY']

    # Optimize allocations
    optimal_allocations = optimize_allocations(prices)

    # Compute portfolio statistics
    cr, adr, sddr, sr = compute_portfolio_stats(prices, optimal_allocations)

    # Generate plot 
    if gen_plot:
        # Normalizing portfolio and SPY for chart
        normed_prices = prices / prices.iloc[0]
        alloced = normed_prices * optimal_allocations
        port_val = alloced.sum(axis=1)
        normalized_SPY = prices_SPY / prices_SPY.iloc[0]
        plot_data(port_val, normalized_SPY)

    return optimal_allocations, cr, adr, sddr, sr
def test_code():
    """
    Test the optimization
    """
    allocations, cr, adr, sddr, sr = optimize_portfolio(gen_plot=True)

if __name__ == "__main__":
    test_code()

