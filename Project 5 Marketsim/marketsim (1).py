""""""  		  	   		 	   			  		 			     			  	 
"""MC2-P1: Market simulator.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
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
import pandas as pd
from util import get_data, plot_data
import datetime as dt                                        
import os                                        

def compute_portvals(
    orders_file="./orders/orders.csv", 
    start_val=1000000, 
    commission=9.95, 
    impact=0.005
):
    # Read in order data
    orders = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan'])
    orders.sort_index(inplace=True)
    start_date, end_date = orders.index.min(), orders.index.max()
    
    # Get date range and symbols
    dates = pd.date_range(start_date, end_date)
    symbols = orders['Symbol'].unique().tolist()
    
    stock_prices = get_data(symbols, dates)
    stock_prices.fillna(method='ffill', inplace=True)  
    stock_prices.fillna(method='bfill', inplace=True)  
    stock_prices['Cash'] = 1.0  
    
    share_tracking = pd.DataFrame(index=stock_prices.index, columns=stock_prices.columns)
    share_tracking.fillna(0.0, inplace=True)  
    share_tracking['Cash'] = start_val  
    
    for date, order in orders.iterrows():
        symbol = order['Symbol']
        shares = order['Shares'] if order['Order'] == 'BUY' else -order['Shares']
        price = stock_prices.loc[date, symbol]

        transaction_cost = shares * price * impact + commission if shares != 0 else 0
        
        share_tracking.loc[date:, symbol] += shares
        share_tracking.loc[date:, 'Cash'] -= (shares * price) + transaction_cost
    
    portfolio_values = (share_tracking * stock_prices).sum(axis=1)
    
    return pd.DataFrame(portfolio_values, columns=['Portfolio Value'])



def author():
    return 'hsahour3' 

def test_code():
    portfolio_values = compute_portvals(
        orders_file="./orders/orders-01.csv",
        start_val=1000000,
        commission=9.95,
        impact=0.005
    )
    daily_returns_portfolio = portfolio_values.pct_change(1).iloc[1:]
    sharpe_ratio_portfolio = daily_returns_portfolio.mean() / daily_returns_portfolio.std() * (252**0.5)
    cumulative_return_portfolio = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
    avg_daily_return_portfolio = daily_returns_portfolio.mean()
    std_daily_return_portfolio = daily_returns_portfolio.std()
    
    start_date = '2008-01-14'
    end_date = '2011-12-14'
    dates = pd.date_range(start_date, end_date)
    spx_data = get_data(['SPY'], dates)['SPY'].to_frame()
    daily_returns_spx = spx_data.pct_change(1).iloc[1:]
    sharpe_ratio_spx = daily_returns_spx.mean() / daily_returns_spx.std() * (252**0.5)
    cumulative_return_spx = (spx_data.iloc[-1] / spx_data.iloc[0]) - 1
    avg_daily_return_spx = daily_returns_spx.mean()
    std_daily_return_spx = daily_returns_spx.std()
    final_portfolio_value = portfolio_values.iloc[-1].values[0]

#     print("Data Range: {} to {}".format(start_date, end_date))
#     print("Sharpe Ratio of Fund: {}".format(sharpe_ratio_portfolio.values[0]))
#     print("Sharpe Ratio of $SPX: {}".format(sharpe_ratio_spx.values[0]))
#     print("Cumulative Return of Fund: {}".format(cumulative_return_portfolio.values[0]))
#     print("Cumulative Return of $SPX: {}".format(cumulative_return_spx.values[0]))
#     print("Standard Deviation of Fund: {}".format(std_daily_return_portfolio.values[0]))
#     print("Standard Deviation of $SPX: {}".format(std_daily_return_spx.values[0]))
#     print("Average Daily Return of Fund: {}".format(avg_daily_return_portfolio.values[0]))
#     print("Average Daily Return of $SPX: {}".format(avg_daily_return_spx.values[0]))
#     print("Final Portfolio Value: {:.2f}".format(final_portfolio_value))
#    plot_data(portfolio_values, title="Portfolio Value Over Time", xlabel="Date", ylabel="Portfolio Value")


if __name__ == "__main__":
    test_code()