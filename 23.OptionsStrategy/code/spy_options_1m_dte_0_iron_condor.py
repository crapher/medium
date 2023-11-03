import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from tqdm import tqdm

# Constants
DEBUG = 1
OPTIONS_FILE='../data/spy_dte_0_iron_condor.csv.gz'

FEES_PER_CONTRACT = 0.6
CONTRACTS = 10

HOUR_OPEN = 11
MINUTE_OPEN = 45
STRIKES = 1

### Helper functions ###
def get_iron_condor_strikes(df, strikes):

    put_sell  = int(df['close_underlying'].values[0]) - strikes
    call_sell = int(df['close_underlying'].values[0]) + strikes
    put_buy   = put_sell - 1
    call_buy  = call_sell + 1

    return (put_buy, put_sell, call_sell, call_buy)

def get_iron_condor_cost(df, put_buy, put_sell, call_sell, call_buy):

    # Open Iron Condor
    df = df.sort_values('date', ascending=True)

    stock_price     = df[(df['strike'] == put_buy)   & (df['kind'] == 'P')]['close_underlying'].values[0]
    put_buy_price   = df[(df['strike'] == put_buy)   & (df['kind'] == 'P')]['close'].values[0]
    put_sell_price  = df[(df['strike'] == put_sell)  & (df['kind'] == 'P')]['close'].values[0]
    call_sell_price = df[(df['strike'] == call_sell) & (df['kind'] == 'C')]['close'].values[0]
    call_buy_price  = df[(df['strike'] == call_buy)  & (df['kind'] == 'C')]['close'].values[0]

    open_cost  = -100 * put_buy_price
    open_cost +=  100 * put_sell_price
    open_cost +=  100 * call_sell_price
    open_cost += -100 * call_buy_price
    open_cost *= CONTRACTS

    if DEBUG:
        tqdm.write(f' OPEN ({stock_price}): {put_buy} @ {put_buy_price} - {put_sell} @ {put_sell_price} - {call_sell} @ {call_sell_price} - {call_buy} @ {call_buy_price}')

    # Close Iron Condor
    stock_price     = df[(df['strike'] == put_buy)   & (df['kind'] == 'P')]['close_underlying'].values[-1]
    put_buy_price   = df[(df['strike'] == put_buy)   & (df['kind'] == 'P')]['close'].values[-1]
    put_sell_price  = df[(df['strike'] == put_sell)  & (df['kind'] == 'P')]['close'].values[-1]
    call_sell_price = df[(df['strike'] == call_sell) & (df['kind'] == 'C')]['close'].values[-1]
    call_buy_price  = df[(df['strike'] == call_buy)  & (df['kind'] == 'C')]['close'].values[-1]

    if DEBUG:
        tqdm.write(f'CLOSE ({stock_price}): {put_buy} @ {put_buy_price} - {put_sell} @ {put_sell_price} - {call_sell} @ {call_sell_price} - {call_buy} @ {call_buy_price}')

    close_cost  =  100 * put_buy_price
    close_cost += -100 * put_sell_price
    close_cost += -100 * call_sell_price
    close_cost +=  100 * call_buy_price
    close_cost *= CONTRACTS

    if DEBUG:
        tqdm.write(f'**************** Processed Date: {df["date"].dt.date.values[0]} - Result: {open_cost + close_cost:7.2f} ****************')

    return open_cost + close_cost

### Read File ###
df = pd.read_csv(OPTIONS_FILE, header=0)
df['date'] = pd.to_datetime(df['date'])

### Get all the expiration dates ###
expirations = df['expire_date'].unique().tolist()

### Get P&L by day ###
pnl = []

df_cost = df[(df.date.dt.hour > HOUR_OPEN) | ((df.date.dt.hour == HOUR_OPEN) & (df.date.dt.minute >= MINUTE_OPEN))]
df_open = df_cost[(df_cost.date.dt.hour == HOUR_OPEN) & (df_cost.date.dt.minute == MINUTE_OPEN)]

for expiration in tqdm(expirations):

    try:
        (put_buy, put_sell, call_sell, call_buy) = get_iron_condor_strikes(df_open[df_open['expire_date'] == expiration], STRIKES)
        cost = get_iron_condor_cost(df_cost[df_cost['expire_date'] == expiration], put_buy, put_sell, call_sell, call_buy)

        pnl.append([expiration, cost])
    except Exception as ex:
        continue

### Generate result ###

# Calculate Net Result
df_result = pd.DataFrame(pnl, columns=['expire_date','gross_result'])
df_result['net_result'] = df_result['gross_result'] - CONTRACTS * FEES_PER_CONTRACT * 4 # Let them expire
df_result['wins'] = np.where(df_result['net_result'] > 0, 1, 0)
df_result['losses'] = np.where(df_result['net_result'] < 0, 1, 0)

# Show the parameters
print(f' PARAMETERS '.center(70, '*'))
print(f'* Hour Open: {HOUR_OPEN}')
print(f'* Minute Open: {MINUTE_OPEN}')
print(f'* Strikes From Price: {STRIKES}')

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
