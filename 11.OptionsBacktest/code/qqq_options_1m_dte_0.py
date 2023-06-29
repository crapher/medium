import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# Configuration
pd.set_option('display.float_format', lambda x: '%.4f' % x)

# Constants
BARS = 15        # Range: 0 - 30
STOP_LOSS = 0.7  # Range: 0 - 1 (0 -> 0% | 1 -> 100%)
POO = 0.01       # Range: 0 - 1 (0 -> 0% | 1 -> 100%)

OPTIONS_FILE='../data/qqq_dte_0.csv.gz'

FEES_PER_CONTRACT = 0.6
CASH = 1000

### Read File ###
df_base = pd.read_csv(OPTIONS_FILE, header=0)
df_base['date'] = pd.to_datetime(df_base['date'])

### Get the trend of each day to see which option we should buy ###

# Get first bar (To get Underlying Open Price)
df_day_open = df_base[(df_base['date'].dt.hour == 9) & (df_base['date'].dt.minute == 30)]

# Get *BARS* bar (To get Underlying Close Price)
df = df_base[(df_base['date'].dt.hour == 9) & (df_base['date'].dt.minute == 30 + BARS - 1)]

# Calculate the trend
df = df.merge(df_day_open,
              how='left',
              left_on=['expire_date','strike','kind'],
              right_on=['expire_date','strike','kind'],
              suffixes=('','_dayopen'))

df.loc[:,'trend'] = np.where(df['open_underlying_dayopen'] < df['close_underlying'], 1,
                    np.where(df['open_underlying_dayopen'] > df['close_underlying'], -1,
                    np.NaN))

# Keep the first open value for each strike
df = df.rename(columns={'open_dayopen': 'option_open'})

# Remove all previous merged values for trend calculation and rows with NaN values
df = df.loc[:,~df.columns.str.endswith('_dayopen')]
df = df.dropna()

### Get the closest ITM option ###

# Filter all puts when trend is going down and calls when trend is going up
df = df[((df['kind'] == 'P') & (df['trend'] == -1)) |
        ((df['kind'] == 'C') & (df['trend'] == 1))]

# Calculate Strike distance from Underlying price
df['distance'] = df['trend'] * (df['close_underlying'] - df['strike'])

# Remove OTM & ATM Options
df = df[df['distance'] > 0]

# Get the closest ITM options
idx = df.groupby(['expire_date'])['distance'].transform(min) == df['distance']
df = df[idx]

# Remove distance column
df = df.drop('distance', axis=1)

### Calculate close points ###

# Get trade bars
df_trade = df_base[((df_base['date'].dt.hour == 9) & (df_base['date'].dt.minute > 30 + BARS - 1)) |
                    (df_base['date'].dt.hour >= 10)]

# Get Option Open and Close Points
df = df_trade.merge(df[['expire_date','kind','strike','option_open']],
                    how='right',
                    left_on=['expire_date','kind','strike'],
                    right_on=['expire_date','kind','strike'])

df.loc[:,'open_point'] = np.where((df['open'] >= df['option_open'] * (1 + POO)) &
                                  ((df['open'].shift() < df['option_open'].shift() * (1 + POO)) |
                                   (df['expire_date'] != df['expire_date'].shift())), 1, 0)

df.loc[:,'stop_loss'] = df['option_open'] * STOP_LOSS
df.loc[:,'last_date'] = df.groupby(['expire_date','kind','strike'])['date'].transform('last')

df.loc[:,'close_point'] = np.where(((df['close'] <= df['stop_loss']) &
                                    ((df['close'].shift() > df['stop_loss'].shift()) | (df['expire_date'] != df['expire_date'].shift()))) |
                                   (df['last_date'] == df['date']), 1, 0)

df_tmp = df[(df['open_point'] - df['close_point']) == 0]
df = df[(df['open_point'] - df['close_point']) != 0]
df.loc[:,'open_point'] = np.where((df['open_point'] - df['close_point']) == (df['open_point'].shift(-1) - df['close_point'].shift(-1)), 0, df['open_point'])

df = pd.concat([df, df_tmp])
df = df.sort_values(by=['date','expire_date','kind','strike'])

# Get Open Price, Close Price, Open Date, and Close Date
df = df[(df['open_point'] != 0) | (df['close_point'] != 0)]

df['open_price'] = np.where(df['open_point'] == 1, df['open'], np.NaN)
df['close_price'] = np.where(df['close_point'] == 1, df['close'], np.NaN)
df['close_price'] = df['close_price'].fillna(method='bfill', limit=1)

df['close_date'] = np.where(df['open_point'] - df['close_point'] == 0, df['date'], df['date'].shift(-1))
df = df.rename(columns={'date':'open_date'})

# Clean all Rows with NaN values (This is going to remove all invalid closes)
df = df.dropna()

# Clean all the unneeded columns
df = df.drop(['last_date','open_point','close_point','open','close'], axis=1)
df = df.loc[:,~df.columns.str.endswith('_underlying')]

# Save the trigger of the closing
df.loc[:,'trigger'] = np.where(df['close_price'] <= df['stop_loss'], 'SL', 'EXPIRED')

### Generate result ###

# Calculate the variables required in the result
df['contracts'] = (CASH // (100 * df['open_price'])).astype(int)
df['fees'] = np.where(df['trigger'] == 'EXPIRED', FEES_PER_CONTRACT, 2 * FEES_PER_CONTRACT) * df['contracts']
df['gross_result'] = df['contracts'] * 100 * (df['close_price'] - df['open_price'])
df['net_result'] = df['gross_result'] - df['fees']

sl = len(df[df["trigger"] == "SL"])
exp = len(df[df["trigger"] != "SL"])
total = len(df)

# Configuration
print(f' CONFIGURATION '.center(70, '*'))
print(f'* Bars: {BARS}')
print(f'* Stop Loss: {STOP_LOSS * 100:.0f}%')
print(f'* Percentage over price: {POO * 100:.0f}%')

# Show the Total Result
print(f' SUMMARIZED RESULT '.center(70, '*'))
print(f'* Trading Days: {len(df["expire_date"].unique())}')
print(f'* Operations: {len(df)} - Stop Loss: {sl} ({100 * sl / total:.2f}%) - Expired: {exp} ({100 * exp / total:.2f}%)')
print(f'* Gross PnL: $ {df["gross_result"].sum():.2f}')
print(f'* Net PnL: $ {df["net_result"].sum():.2f}')

# Show The Monthly Result
print(f' MONTHLY DETAIL RESULT '.center(70, '*'))
df_monthly = df[['expire_date','gross_result','net_result']]
df_monthly['year_month'] = df_monthly['expire_date'].str[0:7]
df_monthly = df_monthly.groupby(['year_month'])[['gross_result','net_result']].sum()
print(df_monthly)
