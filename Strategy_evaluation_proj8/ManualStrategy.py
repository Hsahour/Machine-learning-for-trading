import pandas as pd
import numpy as np
import datetime as dt
from indicators import calculate_simple_moving_average, calculate_rsi, calculate_macd, calculate_bollinger_bands
from marketsimcode import compute_portvals
from util import get_data
import matplotlib.pyplot as plt

class ManualStrategy:
    def __init__(self, verbose=False, impact=0.005, commission=9.95):
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        self.start_val = 100000

    def testPolicy(self, symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):
        dates = pd.date_range(sd, ed)
        prices = get_data([symbol], dates)[[symbol]]
        prices.fillna(method='ffill', inplace=True)
        prices.fillna(method='bfill', inplace=True)
        
        sma = calculate_simple_moving_average(prices[symbol], window=20)
        rsi = calculate_rsi(prices[symbol], window=14)
        macd, signal = calculate_macd(prices[symbol])
        upper_band, _, lower_band = calculate_bollinger_bands(prices[symbol], window=20)
        
        trade_signals = pd.DataFrame(index=prices.index, columns=[symbol], data=0)
        current_position = 0
        
        for i in range(1, len(prices)):
            long_entry_condition = (prices[symbol].iloc[i] < lower_band.iloc[i]) and (rsi.iloc[i] < 30) and (macd.iloc[i] < signal.iloc[i])
            short_entry_condition = (prices[symbol].iloc[i] > upper_band.iloc[i]) and (rsi.iloc[i] > 70) and (macd.iloc[i] > signal.iloc[i])
            
            if long_entry_condition and current_position != 1000:
                trade_signals[symbol].iloc[i] = 1000 - current_position
                current_position = 1000
            elif short_entry_condition and current_position != -1000:
                trade_signals[symbol].iloc[i] = -1000 - current_position
                current_position = -1000
        
        return trade_signals

    def calculate_statistics(self, portfolio_values):
        daily_returns = portfolio_values.pct_change().dropna()
        cumulative_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        std_dev_daily_returns = daily_returns.std()
        mean_daily_returns = daily_returns.mean()
        return {
            "Cumulative Return": cumulative_return,
            "STD DEV of Daily Returns": std_dev_daily_returns,
            "Mean of Daily Returns": mean_daily_returns
        }

    def plot_performance(self, trades, symbol, benchmark_symbol, title, sd, ed, filename):
        dates = pd.date_range(sd, ed)
        prices = get_data([symbol], dates)[[symbol]]
        prices_normalized = prices / prices.iloc[0]

        port_vals = compute_portvals(trades, start_val=self.start_val, commission=self.commission, impact=self.impact)
        port_vals_normalized = port_vals / port_vals.iloc[0]

        benchmark_trades = pd.DataFrame(data={benchmark_symbol: [1000] + [0] * (len(trades) - 1)}, index=trades.index)
        benchmark_vals = compute_portvals(benchmark_trades, start_val=self.start_val, commission=self.commission, impact=self.impact)
        benchmark_normalized = benchmark_vals / benchmark_vals.iloc[0]

        plt.figure(figsize=(14, 7))
        plt.plot(port_vals_normalized.index, port_vals_normalized, label=f'Manual Strategy ({symbol})', color='red')
        plt.plot(benchmark_normalized.index, benchmark_normalized, label=f'Benchmark ({benchmark_symbol})', color='purple')

        long_entries = trades[trades[symbol] > 0].index
        short_entries = trades[trades[symbol] < 0].index
        for long_entry in long_entries:
            plt.axvline(x=long_entry, color='blue', linestyle='--')
        for short_entry in short_entries:
            plt.axvline(x=short_entry, color='black', linestyle='--')

        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Normalized Portfolio Value")
        plt.legend()
        plt.savefig(filename)
        plt.close() 


    def author():
        return 'hsahour3'

if __name__ == "__main__":
    ms = ManualStrategy(verbose=True)
    in_sample_trades = ms.testPolicy(sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31))
    out_of_sample_trades = ms.testPolicy(sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31))
    
    in_sample_port_vals = compute_portvals(in_sample_trades, start_val=ms.start_val, commission=ms.commission, impact=ms.impact)
    out_of_sample_port_vals = compute_portvals(out_of_sample_trades, start_val=ms.start_val, commission=ms.commission, impact=ms.impact)

    ms.plot_performance(in_sample_trades, 'JPM', 'JPM', "In-Sample_manual: Manual Strategy vs. Benchmark", dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31), 'in_sample_performance.png')
    ms.plot_performance(out_of_sample_trades, 'JPM', 'JPM', "Out-of-Sample Manual: Manual Strategy vs. Benchmark", dt.datetime(2010, 1, 1), dt.datetime(2011, 12, 31), 'out_of_sample_performance.png')
