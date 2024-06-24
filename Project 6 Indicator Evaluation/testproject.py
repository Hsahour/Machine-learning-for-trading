import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import TheoreticallyOptimalStrategy as tos
from marketsimcode import compute_portvals
import indicators


def author():
    return 'hsahour'
def test_strategy(symbol="JPM", sd=pd.Timestamp('2008-01-01'), ed=pd.Timestamp('2009-12-31'), sv=100000):
# def test_strategy(symbol="AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000):
    indicators.main()
    trades_optimal = tos.testPolicy(symbol, sd, ed, sv)
    portvals_optimal = compute_portvals(trades_df=trades_optimal, start_val=sv, commission=9.95, impact=0.005)
    portvals_optimal_normalized = portvals_optimal / portvals_optimal.iloc[0]
    portvals_benchmark = tos.compute_benchmark(symbol, sd, ed, sv)
    portvals_benchmark_normalized = portvals_benchmark / portvals_benchmark.iloc[0]
    
    plt.figure(figsize=(14, 7))
    portvals_optimal_normalized.plot(color='red', label='Optimal Portfolio')
    portvals_benchmark_normalized.plot(color='purple', label='Benchmark')
    plt.title("Normalized Portfolio Value - Theoretically Optimal Strategy vs. Benchmark")
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    plt.legend()
    plt.savefig('normalized_comparison_plot.png')
    plt.close()
    
    cum_return_optimal, std_daily_return_optimal, mean_daily_return_optimal = tos.calculate_statistics(portvals_optimal_normalized)
    cum_return_benchmark, std_daily_return_benchmark, mean_daily_return_benchmark = tos.calculate_statistics(portvals_benchmark_normalized)
    
    print("Theoretically Optimal Strategy Statistics:")
    print(f"Cumulative Return: {cum_return_optimal}")
    print(f"Standard Deviation of Daily Returns: {std_daily_return_optimal}")
    print(f"Mean of Daily Returns: {mean_daily_return_optimal}\n")
    
    print("Benchmark Statistics:")
    print(f"Cumulative Return: {cum_return_benchmark}")
    print(f"Standard Deviation of Daily Returns: {std_daily_return_benchmark}")
    print(f"Mean of Daily Returns: {mean_daily_return_benchmark}")

if __name__ == "__main__":
    test_strategy()