import pandas as pd
from util import get_data
import matplotlib.pyplot as plt
from marketsimcode import compute_portvals
import datetime as dt

def author():
    return 'hsahour'



def testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):
    dates = pd.date_range(sd, ed)
    prices_all = get_data([symbol], dates)
    prices = prices_all[symbol]
    trades = pd.DataFrame(0, index=prices.index, columns=[symbol])
    
    for i in range(len(prices)-1):
        if prices[i] < prices[i+1]:
            trades[symbol].iloc[i] = 1000 - trades[symbol].cumsum().iloc[i]  # buy if not already holding
        elif prices[i] > prices[i+1]:
            trades[symbol].iloc[i] = -1000 - trades[symbol].cumsum().iloc[i]  # sell if not already short
    trades[symbol].iloc[-1] = -trades[symbol].cumsum().iloc[-2]  # Exit position on the last day
    
    return trades


def calculate_statistics(portvals):
    daily_returns = portvals.pct_change().dropna()
    cum_return = (portvals[-1] / portvals[0]) - 1
    std_daily_return = daily_returns.std()
    mean_daily_return = daily_returns.mean()
    return cum_return, std_daily_return, mean_daily_return

def compute_benchmark(symbol, sd, ed, sv=100000):
    dates = pd.date_range(sd, ed)
    prices_all = get_data([symbol], dates)
    prices = prices_all[[symbol]]
    trades_benchmark = pd.DataFrame(index=prices.index, data=0, columns=[symbol])
    trades_benchmark.iloc[0] = 1000 
    portvals_benchmark = compute_portvals(trades_df=trades_benchmark, start_val=sv, commission=0, impact=0)
    
    return portvals_benchmark