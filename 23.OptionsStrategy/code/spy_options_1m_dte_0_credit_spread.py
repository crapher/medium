import warnings
import numpy as np
import pandas as pd

from tqdm import tqdm

# Constants
DEBUG = 1
OPTIONS_FILE='../data/spy_dte_0.csv.gz'

FEES_PER_CONTRACT = 0.6
CONTRACTS = 10

HOUR_OPEN = 10
MINUTE_OPEN = 30
STRIKES = 1
DIST_BETWEEN_STRIKES = 2
MAX_CHANGE_BEARISH = -0.35
MIN_CHANGE_BULLISH = 0.35

### Configuration ###
warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)
pd.options.mode.chained_assignment = None

### Helper functions ###
def get_vertical_option_kind(df, max_change_bearish, min_change_bullish):

    changes = 100 * (df['close_underlying'].values / df['close_underlying'].values[0] - 1)

    change = changes[-1]
    min_change = min(changes)
    max_change = max(changes)

    if change < max_change_bearish and max_change < min_change_bullish:
        return 'C' # Sell Call Vertical
    elif change > min_change_bullish and min_change > max_change_bearish:
        return 'P' # Sell Put Vertical
    else:
        return ''

def get_vertical_strikes(df, kind, strikes, dist_between_strikes):

    if kind == 'P':
        sell_strike = int(df['close_underlying'].values[0]) - strikes
        buy_strike  = sell_strike - dist_between_strikes
    else:
        sell_strike = int(df['close_underlying'].values[0]) + strikes
        buy_strike  = sell_strike + dist_between_strikes

    return (buy_strike, sell_strike)

def get_vertical_result(df, kind, buy_strike, sell_strike):

    # Open Vertical
    df = df.sort_values('date', ascending=True)

    stock_price = df['close_underlying'].values[0]
    buy_price   = df[(df['strike'] == buy_strike)  & (df['kind'] == kind)]['close'].values[0]
    sell_price  = df[(df['strike'] == sell_strike) & (df['kind'] == kind)]['close'].values[0]

    open_cost  = -100 * buy_price
    open_cost +=  100 * sell_price
    open_cost *= CONTRACTS

    if DEBUG:
        tqdm.write(f' OPEN {kind} ({stock_price:6.2f}): SELL {sell_strike} @ {sell_price:6.2f} -  BUY {buy_strike} @ {buy_price:6.2f} - TOTAL: {(-buy_price + sell_price):6.2f}')

    # Close Vertical (Prices are 0 if they are OTM/ATM or the difference between the strike and the underlying's close price if they are ITM)
    stock_price = df['close_underlying'].values[-1]
    sell_price = 0 if (kind == 'P' and buy_strike <= stock_price) or (kind == 'C' and buy_strike >= stock_price) else abs(buy_strike - stock_price)
    buy_price = 0 if (kind == 'P' and sell_strike <= stock_price) or (kind == 'C' and sell_strike >= stock_price) else abs(sell_strike - stock_price)

    if DEBUG:
        tqdm.write(f'CLOSE {kind} ({stock_price:6.2f}):  BUY {sell_strike} @ {buy_price:6.2f} - SELL {buy_strike} @ {sell_price:6.2f} - TOTAL: {(-buy_price + sell_price):6.2f}')

    close_cost = -100 * buy_price
    close_cost += 100 * sell_price
    close_cost *= CONTRACTS

    if DEBUG:
        tqdm.write(f'**************** Processed Date: {df["date"].dt.date.values[0]} - Result: {open_cost + close_cost:8.2f} ****************')

    return open_cost + close_cost

### Read File ###
df = pd.read_csv(OPTIONS_FILE, header=0)
df['date'] = pd.to_datetime(df['date'])

### Get all the expiration dates ###
expirations = df['expire_date'].unique().tolist()

### Get P&L by day ###
pnl = []
df_trend = df[(df.date.dt.hour < HOUR_OPEN) | ((df.date.dt.hour == HOUR_OPEN) & (df.date.dt.minute < MINUTE_OPEN))]
df_cost = df[(df.date.dt.hour > HOUR_OPEN) | ((df.date.dt.hour == HOUR_OPEN) & (df.date.dt.minute >= MINUTE_OPEN))]
df_open = df_cost[(df_cost.date.dt.hour == HOUR_OPEN) & (df_cost.date.dt.minute == MINUTE_OPEN)]

for expiration in tqdm(expirations):

    try:
        kind = get_vertical_option_kind(df_trend[df_trend['expire_date'] == expiration], MAX_CHANGE_BEARISH, MIN_CHANGE_BULLISH)

        if kind == '':
            if DEBUG:
                tqdm.write(f'**************** Processed Date: {expiration} - Result: -- NA -- ****************')

            continue

        (buy_strike, sell_strike) = get_vertical_strikes(df_open[df_open['expire_date'] == expiration], kind, STRIKES, DIST_BETWEEN_STRIKES)
        result = get_vertical_result(df_cost[df_cost['expire_date'] == expiration], kind, buy_strike, sell_strike)

        pnl.append([expiration, result])

    except Exception as ex:
        continue

### Generate result ###

# Calculate Net Result
df_result = pd.DataFrame(pnl, columns=['expire_date','gross_result'])
df_result['net_result'] = df_result['gross_result'] - CONTRACTS * FEES_PER_CONTRACT * 2 # Let them expire
df_result['wins'] = np.where(df_result['net_result'] > 0, 1, 0)
df_result['losses'] = np.where(df_result['net_result'] < 0, 1, 0)

# Show the parameters
print(f' PARAMETERS '.center(70, '*'))
print(f'* Hour Open: {HOUR_OPEN}')
print(f'* Minute Open: {MINUTE_OPEN}')
print(f'* Strikes From Price: {STRIKES}')
print(f'* Distance Between Strikes: {DIST_BETWEEN_STRIKES}')
print(f'* Max % Change To Open A Bearish Trade: {MAX_CHANGE_BEARISH}')
print(f'* Min % Change To Open A Bullish Trade: {MIN_CHANGE_BULLISH}')

# Show the Total Result
print(f' SUMMARIZED RESULT '.center(70, '*'))
print(f'* Trading Days: {len(df_result)}')
print(f'* Gross PnL: $ {df_result["gross_result"].sum():.2f}')
print(f'* Net PnL: $ {df_result["net_result"].sum():.2f}')
print(f'* Win Rate: {100 * (df_result["wins"].sum() / (df_result["wins"].sum() + df_result["losses"].sum())):.2f} %')

# Show The Monthly Result
print(f' MONTHLY DETAIL RESULT '.center(70, '*'))
df_monthly = df_result[['expire_date','gross_result','net_result','wins','losses']]
df_monthly['year_month'] = df_monthly['expire_date'].str[0:7]
df_monthly = df_monthly.drop('expire_date', axis=1)
df_monthly = df_monthly.groupby(['year_month']).sum()

print(df_monthly)
