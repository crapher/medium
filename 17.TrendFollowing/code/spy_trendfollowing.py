import sys
import platform

import pandas as pd
import numpy as np

# Constants
CASH = 10000                 # Cash in account

STOP_LOSS_PERC = -2.0        # Maximum allowed loss
TRAILING_STOP = -1.0         # Value percentage for trailing_stop
TRAILING_STOP_TRIGGER = 2.0  # Percentage to start using the trailing_stop to "protect" earnings

GREEN_BARS_TO_OPEN = 4       # Green bars required to open a new position

FILE_NAME = '../data/spy.csv.gz'

# Read File
df = pd.read_csv(FILE_NAME)
df['date'] = pd.to_datetime(df['date'])

# Calculate consecutive bars in the same direction
df['bar_count'] = ((df['open'] < df['close']) != (df['open'].shift() < df['close'].shift())).cumsum()
df['bar_count'] = df.groupby(['bar_count'])['bar_count'].cumcount() + 1
df['bar_count'] = df['bar_count'] * np.where(df['open'].values < df['close'].values,1,-1)

# Variables Initialization
cash = CASH
shares = 0

last_bar = None

operation_last = 'WAIT'

ts_trigger = 0
sl_price = 0

# Generate operations
for index, row in df.iterrows():
    
    # If there is no operation
    if operation_last == 'WAIT':
        if row['close'] == 0:
            continue

        if last_bar is None:
            last_bar = row
            continue

        if row['bar_count'] >= GREEN_BARS_TO_OPEN:

            operation_last = 'LONG'

            open_price = row['close']
            ts_trigger = open_price * (1 + (TRAILING_STOP_TRIGGER / 100))
            sl_price = open_price * (1 + (STOP_LOSS_PERC / 100))

            shares = int(cash // open_price)
            cash -= shares * open_price

        else:
            last_bar = None
            continue
    
    # If the last operation was a purchase
    elif operation_last == 'LONG':

        if row['close'] < sl_price:
            operation_last = 'WAIT'

            cash += shares * row['close']
            shares = 0

            open_price = 0
            ts_trigger = 0
            sl_price = 0

        elif open_price < row['close']:
            if row['close'] > ts_trigger:
                sl_price_tmp = row['close'] * (1 + (TRAILING_STOP / 100))

                if sl_price_tmp > sl_price:
                    sl_price = sl_price_tmp

    print(f"{operation_last:<5}: {round(open_price, 2):8} - Cash: {round(cash, 2):8} - Shares: {shares:4} - CURR PRICE: {round(row['close'], 2):8} ({index}) - CURR POS: {round(shares * row['close'], 2)}")
    last_bar = row

if shares > 0:

    cash += shares * last_bar['close']
    shares = 0

    open_price = 0

print(' RESULT '.center(76, '*'))
print(f"Cash after Trade: {round(cash, 2):8}")
