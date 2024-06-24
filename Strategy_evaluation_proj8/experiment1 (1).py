import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from ManualStrategy import ManualStrategy
from StrategyLearner import StrategyLearner
from util import get_data
from marketsimcode import compute_portvals
import random
def experiment1():
    symbol = 'JPM'
    start_val = 100000
    commission = 9.95
    impact = 0.005
    random.seed(903941641)

    in_sample_start_date = dt.datetime(2008, 1, 1)
    in_sample_end_date = dt.datetime(2009, 12, 31)
    out_sample_start_date = dt.datetime(2010, 1, 1)
    out_sample_end_date = dt.datetime(2011, 12, 31)

    dates = pd.date_range(in_sample_start_date, out_sample_end_date)
    prices = get_data([symbol], dates, addSPY=False).dropna()

    manual_strategy = ManualStrategy(verbose=True, commission=commission, impact=impact)
    strategy_learner = StrategyLearner(verbose=False, impact=impact, commission=commission)

    strategy_learner.add_evidence(symbol=symbol, sd=in_sample_start_date, ed=in_sample_end_date, sv=start_val)

    in_sample_trades_manual = manual_strategy.testPolicy(symbol=symbol, sd=in_sample_start_date, ed=in_sample_end_date, sv=start_val)
    out_sample_trades_manual = manual_strategy.testPolicy(symbol=symbol, sd=out_sample_start_date, ed=out_sample_end_date, sv=start_val)

    in_sample_trades_strategy = strategy_learner.testPolicy(symbol=symbol, sd=in_sample_start_date, ed=in_sample_end_date, sv=start_val)
    out_sample_trades_strategy = strategy_learner.testPolicy(symbol=symbol, sd=out_sample_start_date, ed=out_sample_end_date, sv=start_val)

    def compute_normalized(port_vals):
        if port_vals is not None:
            return port_vals / port_vals.iloc[0]
        return None

    benchmark_trades = pd.DataFrame(data={symbol: [1000] + [0] * (len(prices) - 1)}, index=prices.index)
    benchmark_vals = compute_normalized(compute_portvals(benchmark_trades, start_val=start_val, commission=commission, impact=impact))
    port_vals_manual_in = compute_normalized(compute_portvals(in_sample_trades_manual, start_val=start_val, commission=commission, impact=impact))
    port_vals_manual_out = compute_normalized(compute_portvals(out_sample_trades_manual, start_val=start_val, commission=commission, impact=impact))
    port_vals_strategy_in = compute_normalized(compute_portvals(in_sample_trades_strategy, start_val=start_val, commission=commission, impact=impact))
    port_vals_strategy_out = compute_normalized(compute_portvals(out_sample_trades_strategy, start_val=start_val, commission=commission, impact=impact))

    plt.figure(figsize=(14, 7))
    plt.plot(port_vals_manual_in.index, port_vals_manual_in, label='Manual Strategy In-Sample')
    plt.plot(port_vals_strategy_in.index, port_vals_strategy_in, label='Strategy Learner In-Sample')
    plt.plot(benchmark_vals[in_sample_start_date:in_sample_end_date].index, benchmark_vals[in_sample_start_date:in_sample_end_date], label='Benchmark In-Sample')
    plt.legend()
    plt.title('In-Sample Performance Comparison')
    plt.xlabel('Date')
    plt.ylabel('Normalized Portfolio Value')
    plt.savefig('in_sample_experiment1.png') 
    plt.close()

    plt.figure(figsize=(14, 7))
    plt.plot(port_vals_manual_out.index, port_vals_manual_out, label='Manual Strategy Out-Sample')
    plt.plot(port_vals_strategy_out.index, port_vals_strategy_out, label='Strategy Learner Out-Sample')
    plt.plot(benchmark_vals[out_sample_start_date:out_sample_end_date].index, benchmark_vals[out_sample_start_date:out_sample_end_date], label='Benchmark Out-Sample')
    plt.legend()
    plt.title('Out-of-Sample Performance')
    plt.xlabel('Date')
    plt.savefig('out_of_sample_experiment1.png')  
    plt.close()

def author():
    return 'hsahour3' 

if __name__ == "__main__":
    experiment1()
