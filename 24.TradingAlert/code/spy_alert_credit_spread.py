import sys
import logging
import warnings
import math
import time
import asyncio
import pytz
import pandas as pd
import yfinance as yf

from datetime import datetime, timedelta
from telegram.ext import ApplicationBuilder

### Constants ###
OPEN_TIME = '10:30'

UNDERLYING_TICKER = 'SPY'
ROUND_UNDERLYING_PRICE = 1
STRIKES = 2
DIST_BETWEEN_STRIKES = 1

MAX_CHANGE_BEARISH = -0.35
MIN_CHANGE_BULLISH = 0.35
MIN_PERCENTAGE = 0.2

TOKEN = [YOUR_TOKEN]
CHAT_ID = [YOUR_CHAT_ID]

### Calculated Values ###
CLOSE_TREND_TIME = datetime.strptime(OPEN_TIME, '%H:%M') - datetime.strptime('00:01', '%H:%M')
CLOSE_TREND_TIME = datetime.utcfromtimestamp(CLOSE_TREND_TIME.total_seconds()).strftime('%H:%M')

### Configuration ###
logging.basicConfig(level=logging.INFO, force=True)

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None

loop = asyncio.get_event_loop()
app = None

### Bot functions ###
def create_and_connect_bot():

    global loop, app

    app = ApplicationBuilder().token(TOKEN).build()

    loop.run_until_complete(app.initialize())
    loop.run_until_complete(app.updater.start_polling())
    loop.run_until_complete(app.start())

def send_message_bot(message):

    global app

    message_to_send = f'{UNDERLYING_TICKER} - {datetime.now().date()} - {message}'
    logging.info(f'send_message_bot() -> {message_to_send})')

    loop.run_until_complete(app.bot.send_message(chat_id=CHAT_ID, text=message_to_send))

### Market data functions ###
def get_stock_data():

    # Download stock data
    df = yf.download(tickers=UNDERLYING_TICKER, interval='1m')

    # Remove timezone
    df.index = pd.to_datetime(df.index.strftime('%Y-%m-%d %H:%M:%S'))

    # Filter only regular market time
    df = df[df.index >= pd.Timestamp('today').floor('D')]
    df = df.between_time('9:30', '15:59')

    # Prepare required data
    df = df.reset_index()
    df = df[['Datetime','Close']]
    df.columns = ['date','close']

    logging.debug('get_stock_data() -> Prepared data')
    logging.debug(df)

    return df.sort_values(['date'])

def get_option_data(kind, buy_strike, sell_strike):

    # Download options data
    ticker = yf.Ticker(UNDERLYING_TICKER)
    df = ticker.option_chain(str(datetime.utcnow().date()))

    # Filter option kind
    df = df.puts if kind == 'P' else df.calls

    # Filter option strikes
    df = df[(df['strike'] == buy_strike) | (df['strike'] == sell_strike)]
    df = df[['lastTradeDate','strike','lastPrice']]
    df.columns = ['date','strike','close']

    # Convert date to Eastern Time
    df['date'] = pd.to_datetime(df['date'].dt.tz_convert('US/Eastern').dt.strftime('%Y-%m-%d %H:%M:%S'))

    # Reset index
    df = df.reset_index(drop=True)

    logging.debug('get_option_data() -> Prepared data')
    logging.debug(df)

    return df

### Strategy functions ###
def get_vertical_option_kind(df):

    changes = 100 * (df['close'].values / df['close'].values[0] - 1)
    change = changes[-1]

    min_change = min(changes)
    max_change = max(changes)

    logging.info(f'get_vertical_option_kind() -> Date range: {df["date"].iloc[0]} - {df["date"].iloc[-1]}')
    logging.info(f'get_vertical_option_kind() -> Max change bearish: {MAX_CHANGE_BEARISH} - Min change bullish: {MIN_CHANGE_BULLISH}')
    logging.info(f'get_vertical_option_kind() -> Change from first bar: {change:.2f} | Change (min): {min_change:.2f} | Change (max): {max_change:.2f}')

    if change < MAX_CHANGE_BEARISH and max_change < MIN_CHANGE_BULLISH:
        result = 'C' # Sell Call Vertical
        logging.info(f'get_vertical_option_kind() -> Option kind used to sell vertical: CALL')

    elif change > MIN_CHANGE_BULLISH and min_change > MAX_CHANGE_BEARISH:
        result = 'P' # Sell Put Vertical
        logging.info(f'get_vertical_option_kind() -> Option kind used to sell vertical: PUT')

    else:
        result = ''
        logging.info(f'get_vertical_option_kind() -> Option kind used to sell vertical: UNDETERMINED')

    return result

def get_vertical_strikes(df, kind):

    if kind == 'P':
        rounded_price = int(df['close'].values[0] - (df['close'].values[0] % ROUND_UNDERLYING_PRICE))
        logging.info(f'get_vertical_strikes() -> Underlying Price: [{df["close"].values[0]:.2f}] - Rounded Price: [{rounded_price}]')

        sell_strike = rounded_price - STRIKES
        buy_strike  = sell_strike - DIST_BETWEEN_STRIKES
        logging.info(f'get_vertical_strikes() -> Sell Strike: [{sell_strike}] - Buy Strike: [{buy_strike}]')

    elif kind == 'C':
        rounded_price = int(df['close'].values[0] + (ROUND_UNDERLYING_PRICE - df['close'].values[0]) % ROUND_UNDERLYING_PRICE)
        logging.info(f'get_vertical_strikes() -> Underlying Price: [{df["close"].values[0]:.2f}] - Rounded Price: [{rounded_price}]')

        sell_strike = rounded_price + STRIKES
        buy_strike  = sell_strike + DIST_BETWEEN_STRIKES
        logging.info(f'get_vertical_strikes() -> Sell Strike: [{sell_strike}] - Buy Strike: [{buy_strike}]')

    else:
        raise Exception("get_vertical_strikes() -> Invalid option kind")

    return (buy_strike, sell_strike)

def notify_result(df_stock, df_options, option_kind, buy_strike, sell_strike):

    global app

    stock_price = df_stock['close'].values[-1]
    buy_price   = df_options[(df_options['strike'] == buy_strike)]['close'].values[-1]
    sell_price  = df_options[(df_options['strike'] == sell_strike)]['close'].values[-1]

    logging.info(f'notify_result() -> Underlying Price: [{stock_price:.2f}] - Buy Price: [{buy_price:.2f}] - Sell Price: [{sell_price:.2f}]')
    logging.info(f'notify_result() -> Credit Spread: [{(-buy_price + sell_price):.2f}] - Full Price: [{abs(buy_strike - sell_strike):.2f}]')
    logging.info(f'notify_result() -> Percentage of Full Price: [{(-buy_price + sell_price) / abs(buy_strike - sell_strike):.2f}] - Min Required Percentage: [{MIN_PERCENTAGE:.2f}]')

    if (-buy_price + sell_price) / abs(buy_strike - sell_strike) < MIN_PERCENTAGE:
        message = f'The credit spread does not meet the minimum percentage requirement (Current: {(-buy_price + sell_price) / abs(buy_strike - sell_strike):.2f} - Required: {MIN_PERCENTAGE:.2f})'
    else:
        message = f'Sell {option_kind}{sell_strike} @ ${sell_price:.2f} - Buy {option_kind}{buy_strike} @ ${buy_price:.2f} - Credit Spread: ${(sell_price - buy_price):.2f} (Stock: {stock_price:.2f})'

    send_message_bot(message)

def check_and_notify():

    # Get Stock data (Date must be Eastern Time and only regular market time (9:30 to 15:59) must be received)
    df_stock = get_stock_data()

    # Prepare required datasets
    df_trend = df_stock.set_index('date').between_time('9:30', CLOSE_TREND_TIME).reset_index()
    df_trend_check = df_stock.set_index('date').between_time(CLOSE_TREND_TIME, '16:00').reset_index() # To check if there is any trade on or after the CLOSE_TREND_TIME
    df_open = df_stock.set_index('date').between_time(OPEN_TIME, '16:00').reset_index().head(1)  # Check the first trade on or after the OPEN_TIME

    # Check trend and open data
    if len(df_trend_check) == 0:
        logging.info(f'df_trend check: There is not enough data')
        return False

    if len(df_open) != 1:
        logging.info(f'df_open check: Open data size invalid (Received: {len(df_open)} - Expected: 1')
        return False

    # Get Option Kind
    option_kind = get_vertical_option_kind(df_trend)
    if option_kind == '':
        message = 'Option kind is empty because it does not meet the requirement'
        logging.info(f'option_kind check: {message}')
        send_message_bot(message)
        return True

    # Get Vertical Strikes
    (buy_strike, sell_strike) = get_vertical_strikes(df_open, option_kind)

    # Get Option Data
    df_options = get_option_data(option_kind, buy_strike, sell_strike)
    if len(df_options) != 2:
        logging.info(f'df_options check: Options size invalid (Received: {len(df_options)} - Expected: 2)')
        return False

    # Notify result
    notify_result(df_stock, df_options, option_kind, buy_strike, sell_strike)
    return True

### Main function ###
def main():

    create_and_connect_bot()

    last_notification = None
    while True:

        curr_time = datetime.now().astimezone(pytz.timezone('US/Eastern')).replace(microsecond=0)
        trade_start = curr_time.replace(hour=int(OPEN_TIME.split(":")[0]), minute=int(OPEN_TIME.split(":")[1]), second=0, microsecond=0)
        trade_start_next = trade_start + timedelta(days=1)
        after_market_start = curr_time.replace(hour=16, minute=0, second=0, microsecond=0)

        # Check if:
        # - the user was already notified,
        # - today is weekend, or
        # - we are in after market hours
        if last_notification == trade_start or curr_time.weekday() >= 5 or curr_time > after_market_start:
            difference = math.ceil((trade_start_next - curr_time).total_seconds())
            logging.info(f'Sleeping for {timedelta(seconds=difference)} (Checking @ {trade_start_next})')
            time.sleep(difference)
            continue

        # Check if we are in pre-market hours
        if curr_time < trade_start:
            difference = math.ceil((trade_start - curr_time).total_seconds())
            logging.info(f'Sleeping for {timedelta(seconds=difference)} (Checking @ {trade_start})')
            time.sleep(difference)
            continue

        # If we could not notify, retry in 30 seconds
        if not check_and_notify():
            difference = 30
            logging.info(f'Sleeping for {timedelta(seconds=difference)} (Checking @ {curr_time + timedelta(seconds=difference)})')
            time.sleep(difference)
            continue

        last_notification = trade_start

if __name__ == '__main__':
    main()