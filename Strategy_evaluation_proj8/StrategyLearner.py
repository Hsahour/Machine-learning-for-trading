# """"""                                        
# """                                        
# Template for implementing StrategyLearner  (c) 2016 Tucker Balch                                        
                                        
# Copyright 2018, Georgia Institute of Technology (Georgia Tech)                                        
# Atlanta, Georgia 30332                                        
# All Rights Reserved                                        
                                        
# Template code for CS 4646/7646                                        
                                        
# Georgia Tech asserts copyright ownership of this template and all derivative                                        
# works, including solutions to the projects assigned in this course. Students                                        
# and other users of this template code are advised not to share it with others                                        
# or to make it available on publicly viewable websites including repositories                                        
# such as github and gitlab.  This copyright statement should not be removed                                        
# or edited.                                        
                                        
# We do grant permission to share solutions privately with non-students such                                        
# as potential employers. However, sharing with other current or future                                        
# students of CS 7646 is prohibited and subject to being investigated as a                                        
# GT honor code violation.                                        
                                        
# -----do not edit anything above this line---                                        
                                        
# Student Name: Hossein Sahour                                      
# GT User ID: hsahour3                                        
# GT ID: 903941641                                        
# """                                        
                                        
# """"""                                        
# """                                        
# Template for implementing StrategyLearner  (c) 2016 Tucker Balch                                        
                                        
# Copyright 2018, Georgia Institute of Technology (Georgia Tech)                                        
# Atlanta, Georgia 30332                                        
# All Rights Reserved                                        
                                        
# Template code for CS 4646/7646                                        
                                        
# Georgia Tech asserts copyright ownership of this template and all derivative                                        
# works, including solutions to the projects assigned in this course. Students                                        
# and other users of this template code are advised not to share it with others                                        
# or to make it available on publicly viewable websites including repositories                                        
# such as github and gitlab.  This copyright statement should not be removed                                        
# or edited.                                        
                                        
# We do grant permission to share solutions privately with non-students such                                        
# as potential employers. However, sharing with other current or future                                        
# students of CS 7646 is prohibited and subject to being investigated as a                                        
# GT honor code violation.                                        
                                        
# -----do not edit anything above this line---                                        
                                        
# Student Name: Hossein Sahour                                      
# GT User ID: hsahour3                                        
# GT ID: 903941641                                        
# """                                        
                                        
import numpy as np
import pandas as pd
import datetime as dt
from QLearner import QLearner
from marketsimcode import compute_portvals
from util import get_data
from indicators import *
import random

def calculate_statistics(portfolio_values):
    daily_returns = portfolio_values.pct_change().dropna()
    cumulative_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
    std_dev_daily_returns = daily_returns.std()
    mean_daily_returns = daily_returns.mean()
    return {
        "Cumulative Return": cumulative_return,
        "STD DEV of Daily Returns": std_dev_daily_returns,
        "Mean of Daily Returns": mean_daily_returns
    }

class StrategyLearner:
    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        self.learner = QLearner(num_states=100, num_actions=5, alpha=0.5, gamma=0.9, rar=0.2, radr=1.9, dyna=20, verbose=verbose)
        random.seed(903941641)

    def add_evidence(self, symbol, sd, ed, sv):

        dates = pd.date_range(sd, ed)
        prices = get_data([symbol], dates)[symbol]
        high = get_data([symbol], dates, colname="High")[symbol] 
        low = get_data([symbol], dates, colname="Low")[symbol]   

        # Calculate indicators
        sma = calculate_simple_moving_average(prices)
        rsi = calculate_rsi(prices)
        macd, signal = calculate_macd(prices)
        upper_band, sma, lower_band = calculate_bollinger_bands(prices)
        ema = calculate_exponential_moving_average(prices)
        ppo = calculate_percentage_price_Indicator(prices)
        stoch = calculate_stochastic_oscillator(prices, high, low)

        features = pd.DataFrame({
            'sma': sma,
            'rsi': rsi,
            'macd': macd,
            'macd_signal': signal,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'ema': ema,
            'ppo': ppo,
            'stoch_k': stoch['%K'],
            'stoch_d': stoch['%D']
        }).fillna(method='bfill').fillna(method='ffill')

        # Discretize features and train QLearner
        state = self.discretize_features(features.iloc[0])
        self.learner.querysetstate(state)
        for t in range(1, len(features)):
            state_prime = self.discretize_features(features.iloc[t])
            reward = self.compute_reward(prices.iloc[t-1], prices.iloc[t])
            self.learner.query(state_prime, reward)

    def discretize_features(self, features):
        return int(features.sum()) % self.learner.num_states

    def compute_reward(self, price_today, price_tomorrow):
        return (price_tomorrow - price_today) 

    def testPolicy(self, symbol, sd, ed, sv):
        dates = pd.date_range(sd, ed)
        prices = get_data([symbol], dates)[symbol]
        trades = pd.DataFrame(data=np.zeros(len(prices)), index=prices.index, columns=[symbol])
        position = 0

        for t in range(len(prices)):
            state = self.discretize_features(prices.iloc[t])
            action = self.learner.querysetstate(state)

            if action == 1 and position + 1000 <= 1000:  # Buy
                trade = 1000 - position
                position += trade
                trades.iloc[t] = trade
            elif action == 2 and position - 1000 >= -1000:  # Sell
                trade = -1000 - position
                position += trade
                trades.iloc[t] = trade

        return trades

    def calculate_stats(self, trades, symbol, sd, ed, sv):
        dates = pd.date_range(sd, ed)
        prices = get_data([symbol], dates)[symbol]
        port_vals = compute_portvals(trades, start_val=sv, commission=self.commission, impact=self.impact)
        benchmark_trades = pd.DataFrame(data={symbol: [1000] + [0]*(len(prices)-1)}, index=prices.index)
        benchmark_vals = compute_portvals(benchmark_trades, start_val=sv, commission=self.commission, impact=self.impact)
        strategy_stats = calculate_statistics(port_vals)
        benchmark_stats = calculate_statistics(benchmark_vals)
        return strategy_stats, benchmark_stats

    def author(self):
        return 'hsahour3' 
if __name__ == "__main__":
    learner = StrategyLearner(verbose=False, impact=0.005, commission=0.01)
    learner.add_evidence(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    trades = learner.testPolicy(symbol="JPM", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000)
    strategy_stats, benchmark_stats = learner.calculate_stats(trades, symbol="JPM", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000)
#     print(strategy_stats)
#     print(benchmark_stats)



