import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from StrategyLearner import StrategyLearner

def experiment2():
    symbol = 'JPM'
    start_val = 100000
    commission = 0.00
    in_sample_start_date = dt.datetime(2008, 1, 1)
    in_sample_end_date = dt.datetime(2009, 12, 31)
    impacts = [0.0, 0.005, 0.01]  
    
    cumulative_returns = []
    std_devs = []

    for impact in impacts:
        learner = StrategyLearner(verbose=False, impact=impact, commission=commission)
        learner.add_evidence(symbol=symbol, sd=in_sample_start_date, ed=in_sample_end_date, sv=start_val)
        trades = learner.testPolicy(symbol=symbol, sd=in_sample_start_date, ed=in_sample_end_date, sv=start_val)
        strategy_stats, _ = learner.calculate_stats(trades, symbol=symbol, sd=in_sample_start_date, ed=in_sample_end_date, sv=start_val)
        cumulative_returns.append(strategy_stats['Cumulative Return'])
        std_devs.append(strategy_stats['STD DEV of Daily Returns'])
    

    plt.figure(figsize=(10, 5))
    plt.plot(impacts, cumulative_returns, marker='o', color='red', label='Cumulative Return')
    plt.title('Impact vs Cumulative Return')
    plt.xlabel('Impact')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.savefig('impact_vs_cumulative_return.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(impacts, std_devs, marker='o', color='blue', label='STD DEV of Daily Returns')
    plt.title('Impact vs STD DEV of Daily Returns')
    plt.xlabel('Impact')
    plt.ylabel('STD DEV of Daily Returns')
    plt.legend()
    plt.savefig('impact_vs_std_dev.png') 
    plt.close()

#     print("Impact vs Cumulative Return:")
#     for impact, cum_ret in zip(impacts, cumulative_returns):
#         print(f"Impact: {impact}, Cumulative Return: {cum_ret}")

#     print("\nImpact vs STD DEV of Daily Returns:")
#     for impact, std_dev in zip(impacts, std_devs):
#         print(f"Impact: {impact}, STD DEV: {std_dev}")
def author():
    return 'hsahour3' 
if __name__ == "__main__":
    experiment2()
