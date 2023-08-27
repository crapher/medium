import numpy as np
import pandas as pd

# Constants
FILENAME = '../data/qqq_puts.csv.gz'
FEES_PER_CONTRACT = 0.6
CONTRACTS_QTY = 10

# Read file
df = pd.read_csv(FILENAME, header=0)

# Convert date fields to datetime
df['date'] = pd.to_datetime(df['date'])
df['expire_date'] = pd.to_datetime(df['expire_date'])

# Get all the options expiring int 7 days
df_open = df[df['dte'] == 7]

# Get all the expired options
# If the underlying last > strike, set the ask price to 0.00
df_close = df[(df['dte'] == 0)]
df_close.loc[df_close['underlying_last'] >= df_close['strike'], 'ask'] = 0

# Filter options with delta close to 0.18
df_open = df_open[(df_open['delta'] > -0.18) & (df_open['delta'] < -0.10)]
idx = df_open.groupby(['expire_date'])['delta'].transform(min) == df_open['delta']
df_open = df_open[idx]

# Generate the dataset with the combination
df_op = pd.merge(df_open, df_close, how='inner', on=['expire_date','strike'], suffixes=['_sell','_buy'])
df_op = df_op.reset_index(drop=True)

# Calculate fees
df_op['fees'] = np.where(df_op['underlying_last_buy'] >= df_op['strike'], FEES_PER_CONTRACT, 2 * FEES_PER_CONTRACT)

# Generate result
puts_qty = len(df_op)
puts_itm = len(df_op[df_op['underlying_last_buy'] < df_op['strike']])
profit_loss = (((df_op['bid_sell'] - df_op['ask_buy']) * 100 - df_op['fees']) * CONTRACTS_QTY).sum()

print(f' NAKED PUTS STRATEGY - RESULT '.center(70, '*'))
print(f'Closing ITM: {100 * puts_itm / puts_qty:.2f}% ({puts_itm} / {puts_qty})')
print(f"     Result: $ {profit_loss:.2f}")
