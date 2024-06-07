import warnings
import numpy as np
import pandas as pd

from tqdm import tqdm

# Constants
DEBUG = 1
OPTIONS_FILE='../data/spx_dte_0.csv.gz'

# TastyTrade
OPEN_FEES_PER_CONTRACT = 1.00

# Schwab
#OPEN_FEES_PER_CONTRACT = 0.65

CONTRACTS = 10

HOUR_OPEN = 10
MINUTE_OPEN = 30
STRIKES = 15
DIST_BETWEEN_STRIKES = 5

MAX_CHANGE_BEARISH = -0.10
MIN_CHANGE_BULLISH = 0.10
MIN_DAY_CHANGE_BULLISH = -0.15
MAX_DAY_CHANGE_BEARISH = 0.15
MIN_STRIKE_PERCENTAGE = 0.15

### Configuration ###
warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)
pd.options.mode.chained_assignment = None

### Helper functions ###
def get_vertical_option_kind(df, max_change_bearish, min_change_bullish, max_day_change_bearish, min_day_change_bullish):

    changes = 100 * (df['close_underlying'].values / df['close_underlying'].values[0] - 1)
    change = changes[-1]

    min_change = min(changes)
    max_change = max(changes)

    if change < max_change_bearish and max_change < max_day_change_bearish:
        return 'C' # Sell Call Vertical
    elif change > min_change_bullish and min_change > min_day_change_bullish:
        return 'P' # Sell Put Vertical
    else:
        return ''

def get_vertical_strikes(df, kind, strikes, dist_between_strikes):

    strike_up = int(df['close_underlying'].values[0] + (dist_between_strikes - df['close_underlying'].values[0]) % dist_between_strikes)
    strike_down = int(df['close_underlying'].values[0] - (df['close_underlying'].values[0] % dist_between_strikes))

    if kind == 'P':
        sell_strike = strike_down - strikes
        buy_strike  = sell_strike - dist_between_strikes
    else:
        sell_strike = strike_up + strikes
        buy_strike  = sell_strike + dist_between_strikes

    return (buy_strike, sell_strike)

def get_vertical_result(df, kind, buy_strike, sell_strike, min_strike_percentage):

    # Filter strikes
    df = df[(df['kind'] == kind) & ((df['strike'] == buy_strike) | (df['strike'] == sell_strike))]

    # Open Vertical
    df = df.sort_values('date', ascending=True)

    stock_price = df['close_underlying'].values[0]
    buy_price   = df[(df['strike'] == buy_strike)  & (df['kind'] == kind)]['close'].values[0]
    sell_price  = df[(df['strike'] == sell_strike) & (df['kind'] == kind)]['close'].values[0]

    if (-buy_price + sell_price) / abs(buy_strike - sell_strike) < min_strike_percentage:
        return None

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
max_date_trend = df_trend.date.max()
df_open = df_trend[(df_trend.date.dt.hour == max_date_trend.hour) & (df_trend.date.dt.minute == max_date_trend.minute)]
df_cost = df[(df.date.dt.hour > max_date_trend.hour) | ((df.date.dt.hour == max_date_trend.hour) & (df.date.dt.minute >= max_date_trend.minute))]

df_trend = df_trend.groupby(by='expire_date')
df_cost = df_cost.groupby(by='expire_date')
df_open = df_open.groupby(by='expire_date')

for expiration in tqdm(expirations):

    try:
        kind = get_vertical_option_kind(df_trend.get_group(expiration), MAX_CHANGE_BEARISH, MIN_CHANGE_BULLISH, MAX_DAY_CHANGE_BEARISH, MIN_DAY_CHANGE_BULLISH)

        if kind == '':
            if DEBUG:
                tqdm.write(f' Processed Date: {expiration} - Result: -- NA -- '.center(84, '*'))

            continue

        (buy_strike, sell_strike) = get_vertical_strikes(df_open.get_group(expiration), kind, STRIKES, DIST_BETWEEN_STRIKES)
        gross_pnl = get_vertical_result(df_cost.get_group(expiration), kind, buy_strike, sell_strike, MIN_STRIKE_PERCENTAGE)

        if gross_pnl is None:

            if DEBUG:
                tqdm.write(f' Processed Date: {expiration} - Result: -- UV -- '.center(84, '*'))

            continue
        
        net_pnl = gross_pnl - CONTRACTS * OPEN_FEES_PER_CONTRACT * 2
        pnl.append([expiration, gross_pnl, net_pnl])

    except Exception as ex:
        if DEBUG:
            tqdm.write(f' Processed Date: {expiration} - Result: -- ER -- '.center(84, '*'))

### Generate result ###

# Calculate wins and losses
df_result = pd.DataFrame(pnl, columns=['expire_date','gross_result','net_result'])
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
print(f'* Max Day Change To Open A Bearish Trade: {MAX_DAY_CHANGE_BEARISH}')
print(f'* Min Day Change To Open A Bullish Trade: {MIN_DAY_CHANGE_BULLISH}')
print(f'* Min Strike Percentage: {MIN_STRIKE_PERCENTAGE}')

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

# Show The Yearly Result
print(f' YEARLY DETAIL RESULT '.center(70, '*'))
df_yearly = df_result[['expire_date','gross_result','net_result','wins','losses']]
df_yearly['year'] = df_yearly['expire_date'].str[0:4]
df_yearly = df_yearly.drop('expire_date', axis=1)
df_yearly = df_yearly.groupby(['year']).sum()

print(df_yearly)
