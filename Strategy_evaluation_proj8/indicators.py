import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util import get_data, plot_data
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

def get_author():
    return 'hsahour'

def calculate_simple_moving_average(prices, window=12):
    return prices.rolling(window=window, min_periods=1).mean()

def calculate_bollinger_bands(prices, window=12, num_std_dev=2):
    sma = calculate_simple_moving_average(prices, window)
    rolling_std = prices.rolling(window=window, min_periods=1).std()
    upper_band = sma + (rolling_std * num_std_dev)
    lower_band = sma - (rolling_std * num_std_dev)
    return upper_band, sma, lower_band

def calculate_rsi(prices, window=12):
    delta = prices.diff(1)
    gain = delta.where(delta > 0, 0).rolling(window=window, min_periods=1).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window, min_periods=1).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_exponential_moving_average(prices, window=12):
    return prices.ewm(span=window, adjust=False).mean()

def calculate_macd(prices, slow_window=26, fast_window=12, signal_window=9):
    slow_ema = calculate_exponential_moving_average(prices, slow_window)
    fast_ema = calculate_exponential_moving_average(prices, fast_window)
    macd = fast_ema - slow_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

def calculate_percentage_price_Indicator(prices, window=12):
    previous_prices = prices.shift(window)  
    ppo = ((prices - previous_prices) / previous_prices) * 100 
    return ppo

def calculate_stochastic_oscillator(prices, high, low, window=12, smooth_k=3, smooth_d=3):
    lowest_low = low.rolling(window=window, min_periods=window).min()
    highest_high = high.rolling(window=window, min_periods=window).max()
    diff = highest_high - lowest_low
    k_percent = ((prices - lowest_low) / diff.where(diff != 0, np.nan)) * 100
    k_smooth = k_percent.rolling(window=smooth_k, min_periods=smooth_k).mean()
    d_percent = k_smooth.rolling(window=smooth_d, min_periods=smooth_d).mean()
    return pd.DataFrame({'%K': k_smooth, '%D': d_percent})



def main():
    symbols = ['JPM']
    dates = pd.date_range('2008-01-01', '2009-12-31')
    df = get_data(symbols, dates, addSPY=False, colname='Adj Close')
    df['High'] = get_data(symbols, dates, addSPY=False, colname='High')
    df['Low'] = get_data(symbols, dates, addSPY=False, colname='Low')
    df = df.fillna(method='ffill').fillna(method='bfill')
    # Normalize prices for 'JPM'
    prices = df[symbols[0]]
    prices_norm = df / df.iloc[0]
    prices_norm = prices_norm["JPM"]
    # calculate indicators
    sma = calculate_simple_moving_average(prices_norm)
    upper_band, _, lower_band = calculate_bollinger_bands(prices_norm)
    rsi = calculate_rsi(prices_norm)
    macd, macd_signal = calculate_macd(prices_norm)
    ppi = calculate_percentage_price_Indicator(prices, window=12)
    stoch = calculate_stochastic_oscillator(df['JPM'], df['High'], df['Low'])
  
    #Bollinger
    plt.figure(figsize=(14, 7))
    plt.plot(prices.index, prices_norm, label='Normalized Price')
    plt.plot(sma.index, sma, label='SMA')
    plt.plot(upper_band.index, upper_band, label='Upper BB', linestyle='--')
    plt.plot(lower_band.index, lower_band, label='Lower BB', linestyle='--')
    plt.title('Price, SMA and Bollinger Bands')
    plt.xlabel('Date') 
    plt.ylabel('Normalized Price') 
    plt.legend()
    plt.savefig('Bollinger.png')
    plt.close()

    # RSI Plot
    plt.figure(figsize=(14, 7))
    plt.plot(rsi.index, rsi, label='RSI')
    plt.axhline(70, linestyle='--', alpha=0.5, color='red')
    plt.axhline(30, linestyle='--', alpha=0.5, color='green')
    plt.title('Relative Strength Index')
    plt.xlabel('Date') 
    plt.ylabel('RSI Value') 
    plt.legend()
    plt.savefig('RSI.png', dpi=300)    
    plt.close()

    # MACD Plot
    plt.figure(figsize=(14, 7))
    plt.plot(macd.index, macd, label='MACD', color='blue')
    plt.plot(macd_signal.index, macd_signal, label='Signal Line', color='red')
    plt.axhline(0, linestyle='--', color='grey', alpha=0.5) 
    plt.title('Moving Average Convergence Divergence (MACD)')
    plt.xlabel('Date') 
    plt.ylabel('MACD Value')  
    plt.legend()
    plt.savefig('MACD', dpi=300)
    plt.close()

    # Stochastic Oscillator Plot
    plt.figure(figsize=(14, 7))
    plt.plot(stoch.index, stoch['%K'], label='%K line')
    plt.plot(stoch.index, stoch['%D'], label='%D line (Signal)', linestyle='--')
    plt.title('Stochastic Oscillator')
    plt.xlabel('Date')  
    plt.ylabel('Stochastic Value')  
    plt.legend()
    plt.savefig('Stochastic_Oscillator.png')
    plt.close()

    
    # Plot for Percentage Price Indicator
    plt.figure(figsize=(14, 7))
    plt.plot(ppi.index, ppi, label='Percentage Price Indicator', color='green')
    plt.plot(prices.index, prices_norm, label='Price')
    plt.xlabel('Date')
    plt.ylabel('Percentage')
    plt.legend()
    plt.savefig('PPI.png')
    plt.close()
if __name__ == "__main__":
    main()
