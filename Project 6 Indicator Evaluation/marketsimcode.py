import pandas as pd
from util import get_data
import datetime as dt
import os

def author():
    return 'hsahour3'


def compute_portvals(trades_df, start_val=100000, commission=9.95, impact=0.005):
    if trades_df.empty:
        return None
    
    symbols = trades_df.columns.tolist()
    start_date = trades_df.index.min()
    end_date = trades_df.index.max()
    price_data = get_data(symbols, pd.date_range(start_date, end_date), addSPY=True, colname='Adj Close').drop('SPY', axis=1)
    price_data['Cash'] = 1.0
    holdings = pd.DataFrame(index=price_data.index, columns=price_data.columns)
    holdings[:] = 0
    holdings['Cash'] = start_val
    
    for date, trades in trades_df.iterrows():
        for symbol in symbols:
            trade_volume = trades[symbol]
            if trade_volume != 0:
                trade_price = price_data.loc[date, symbol]
                trade_impact = trade_price * impact * abs(trade_volume)
                trade_cost = trade_price * trade_volume + commission + trade_impact
                holdings.loc[date:, symbol] += trade_volume
                holdings.loc[date:, 'Cash'] -= trade_cost
    
    value = holdings * price_data
    portvals = value.sum(axis=1)
    
    return portvals


