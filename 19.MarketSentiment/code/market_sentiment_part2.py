import os
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta

# Configuration
np.set_printoptions(suppress=True)
pd.options.mode.chained_assignment = None

# Constants
SYMBOL_SD = 'E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE'
SYMBOLS_SD_TO_MERGE = ['E-MINI S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE']
SYMBOL_QT = 'SPY'

FILENAME_SD = '../data/market_sentiment_data.csv.gz'
FILENAME_QT = f'../data/{SYMBOL_QT}.csv.gz'

CASH = 10_000
BB_LENGTH = 20
MIN_BANDWIDTH = 0
MAX_BUY_PERC = 0.25
MIN_SELL_PERC = 0.75

def get_data():

    # Read Sentiment Data
    df_sd = pd.read_csv(FILENAME_SD)

    # Merge Symbols If Exists A Symbol With Different Names
    if SYMBOLS_SD_TO_MERGE is not None or len(SYMBOLS_SD_TO_MERGE) > 0:
        for symbol_to_merge in SYMBOLS_SD_TO_MERGE:
            df_sd['Market_and_Exchange_Names'] = df_sd['Market_and_Exchange_Names'].str.replace(symbol_to_merge, SYMBOL_SD)

    # Sort By Report Date
    df_sd = df_sd.sort_values('Report_Date_as_YYYY-MM-DD')

    # Filter Required Symbol
    df_sd = df_sd[df_sd['Market_and_Exchange_Names'] == SYMBOL_SD]
    df_sd['Report_Date_as_YYYY-MM-DD'] = pd.to_datetime(df_sd['Report_Date_as_YYYY-MM-DD'])

    # Remove Unneeded Columns And Rename The Rest
    df_sd = df_sd.rename(columns={'Report_Date_as_YYYY-MM-DD':'report_date'})
    df_sd = df_sd.drop('Market_and_Exchange_Names', axis=1)

    # Read / Get & Save Market Data
    if not os.path.exists(FILENAME_QT):
        ticker = yf.Ticker(SYMBOL_QT)
        df = ticker.history(
            interval='1d',
            start=min(df_sd['report_date']),
            end=max(df_sd['report_date']))

        df = df.reset_index()
        df['Date'] = df['Date'].dt.date
        df = df[['Date','Close']]
        df.columns = ['date', 'close']
        if len(df) > 0: df.to_csv(FILENAME_QT, index=False)
    else:
        df = pd.read_csv(FILENAME_QT)

    df['date'] = pd.to_datetime(df['date'])

    # Merge Market Sentiment Data And Market Data
    tolerance = pd.Timedelta('7 day')
    df = pd.merge_asof(left=df_sd,right=df,left_on='report_date',right_on='date',direction='backward',tolerance=tolerance)
    df = df.rename(columns={'date':'quote_date'})

    # Clean Data And Rename Columns
    df = df.dropna()
    df.columns = ['report_date', 'dealer_long', 'dealer_short', 'lev_money_long', 'lev_money_short', 'quote_date', 'close']

    return df

def get_result(df, field, bb_length, min_bandwidth, max_buy_perc, min_sell_perc):

    # Generate a copy to avoid changing the original data
    df = df.copy().reset_index(drop=True)

    # Calculate Bollinger Bands With The Specified Field
    df.ta.bbands(close=df[field], length=bb_length, append=True)
    df['high_limit'] = df[f'BBU_{bb_length}_2.0'] + (df[f'BBU_{bb_length}_2.0'] - df[f'BBL_{bb_length}_2.0']) / 2
    df['low_limit'] = df[f'BBL_{bb_length}_2.0'] - (df[f'BBU_{bb_length}_2.0'] - df[f'BBL_{bb_length}_2.0']) / 2
    df['close_percentage'] = np.clip((df[field] - df['low_limit']) / (df['high_limit'] - df['low_limit']), 0, 1)
    df['bandwidth'] = np.clip(df[f'BBB_{bb_length}_2.0'] / 100, 0, 1)

    df = df.dropna()

    # Buy Signal
    df['signal'] = np.where((df['bandwidth'] > min_bandwidth) & (df['close_percentage'] < max_buy_perc), 1, 0)

    # Sell Signal
    df['signal'] = np.where((df['close_percentage'] > min_sell_perc), -1, df['signal'])

    # Remove all rows without operations, rows with the same consecutive operation, first row selling, and last row buying
    result = df[df['signal'] != 0]
    result = result[result['signal'] != result['signal'].shift()]
    if (len(result) > 0) and (result.iat[0, -1] == -1): result = result.iloc[1:]
    if (len(result) > 0) and (result.iat[-1, -1] == 1): result = result.iloc[:-1]

    # Calculate the reward / operation
    result['total_reward'] = np.where(result['signal'] == -1, (result['close'] - result['close'].shift()) * (CASH // result['close'].shift()), 0)

    # Generate the result
    total_reward = result['total_reward'].sum()
    wins = len(result[result['total_reward'] > 0])
    losses = len(result[result['total_reward'] < 0])

    return total_reward, wins, losses

# Get Required Data
df = get_data()

# Get Result Based Calculating the BB on Each Field to Check Which is the Most Accurate
for field in ['dealer_long', 'dealer_short', 'lev_money_long', 'lev_money_short']:

    total_reward, wins, losses = get_result(df, field, BB_LENGTH, MIN_BANDWIDTH, MAX_BUY_PERC, MIN_SELL_PERC)

    print(f' Result (Field: {field}) '.center(60, '*'))
    print(f"* Profit / Loss  : {total_reward:.2f}")
    print(f"* Wins / Losses  : {wins} / {losses}")
    print(f"* Win Rate       : {(100 * (wins/(wins + losses)) if wins + losses > 0 else 0):.2f}%")
