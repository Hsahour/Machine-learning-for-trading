import datetime as dt
from ManualStrategy import ManualStrategy
from StrategyLearner import StrategyLearner
import experiment1
import experiment2

def testproject():
    symbol = "JPM"
    start_val = 100000
    commission = 9.95
    impact = 0.005
    in_sample_start_date = dt.datetime(2008, 1, 1)
    in_sample_end_date = dt.datetime(2009, 12, 31)
    out_sample_start_date = dt.datetime(2010, 1, 1)
    out_sample_end_date = dt.datetime(2011, 12, 31)

    manual_strategy = ManualStrategy(verbose=True, impact=impact, commission=commission)
    
    in_sample_trades = manual_strategy.testPolicy(symbol=symbol, sd=in_sample_start_date, ed=in_sample_end_date, sv=start_val)
    out_sample_trades = manual_strategy.testPolicy(symbol=symbol, sd=out_sample_start_date, ed=out_sample_end_date, sv=start_val)
    
    manual_strategy.plot_performance(in_sample_trades, symbol, symbol, "In-Sample Performance: Manual Strategy vs. Benchmark", in_sample_start_date, in_sample_end_date, 'in_sample_performance.png')
    manual_strategy.plot_performance(out_sample_trades, symbol, symbol, "Out-of-Sample Performance: Manual Strategy vs. Benchmark", out_sample_start_date, out_sample_end_date, 'out_sample_performance.png')

    strategy_learner = StrategyLearner(verbose=True, impact=impact, commission=commission)
    strategy_learner.add_evidence(symbol=symbol, sd=in_sample_start_date, ed=in_sample_end_date, sv=start_val)
    strategy_in_sample_trades = strategy_learner.testPolicy(symbol=symbol, sd=in_sample_start_date, ed=in_sample_end_date, sv=start_val)
    strategy_out_sample_trades = strategy_learner.testPolicy(symbol=symbol, sd=out_sample_start_date, ed=out_sample_end_date, sv=start_val)

    experiment1.experiment1()
    experiment2.experiment2()

if __name__ == "__main__":
    testproject()
