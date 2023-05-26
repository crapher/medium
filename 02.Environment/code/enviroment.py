import numpy as np
import pandas as pd
import pandas_ta as ta

TIMEFRAMES = ['15T', '1H', '1D']

# Using this method, you can obtain buy and sell signals determined by the selected strategy.
# The resulting signals are represented as a series of numerical values: 
#   '1' indicating a buy signal, 
#   '0' indicating a hold signal, and 
#   '-1' indicating a sell signal
def get_signals(df):
    
    # Get the MACD values (MACD, Histogram, and Signal) and rename the columns to something meaningful
    macd = df.ta.macd().dropna().reset_index(drop=True)
    macd.columns = ['macd', 'histogram', 'signal']
    
    # Calculate the buy and sell points using the Zero-Cross Strategy
    macd['signal'] = np.where((macd['histogram'] > 0) & (macd['histogram'].shift() <= 0), 1, 0)
    macd['signal'] = np.where((macd['histogram'] < 0) & (macd['histogram'].shift() >= 0), -1, macd['signal'])

    return macd['signal']

# Using this method, you can visualize the results of a simulated long position strategy. 
# Note that it assumes the purchase of one share per transaction and does not account for any fees.
def show_stategy_result(timeframe, df):
    
    waiting_for_close = False
    open_price = 0

    profit = 0.0
    wins = 0
    losses = 0
    
    for i in range(len(df)):
    
        signal = df.iloc[i]['signal'] 
        
        if signal == 1 and not waiting_for_close:
            waiting_for_close = True
            open_price = df.iloc[i]['close']

        elif signal == -1 and waiting_for_close:
            waiting_for_close = False
            close_price = df.iloc[i]['close']
            
            profit += close_price - open_price
            wins = wins + (1 if (close_price - open_price) > 0 else 0)
            losses = losses + (1 if (close_price - open_price) < 0 else 0)

    print(f' Result for timeframe {timeframe} '.center(60, '*'))
    print(f'* Profit/Loss: {profit:.2f}')
    print(f"* Wins: {wins} - Losses: {losses}")
    print(f"* Win Rate: {100 * (wins/(wins + losses)):6.2f}%")
    
# Iterate over each timeframe, apply the strategy and show the result
for timeframe in TIMEFRAMES:

    # Read the data
    df = pd.read_csv(f'/Users/diego/Developer/datasets/finance/stocks/{timeframe}_cdd/OIH.csv.gz', compression='gzip')
#    df = pd.read_csv(f'OIH_{timeframe}.csv.gz', compression='gzip')
    
    # Add the signals to each row
    df['signal'] = get_signals(df)
    
    # Get the result of the strategy
    show_stategy_result(timeframe, df)