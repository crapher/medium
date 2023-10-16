import sys
import os
import ccxt
import pandas as pd

from tqdm import tqdm
from datetime import datetime
from time import sleep

# Constants
TIME_FRAME = '5m'
START_DATE = datetime.fromtimestamp(int(datetime.now().timestamp() - (180 * 86400))).strftime("%Y-%m-%d")
END_DATE = int((datetime.now().timestamp() - 86400) * 1000)

RESULT_DIR = '../data'
QUOTE_CURRENCY = 'USDT'
IS_USMARKET = True

# Create result folder
os.makedirs(RESULT_DIR, exist_ok=True)

# Define the Exchange Connector
exchange = ccxt.binanceus({'enableRateLimit': True}) if IS_USMARKET else ccxt.binance({'enableRateLimit': True})

# Getting the Available Trading Pairs (Markets)
markets = exchange.load_markets()
    
# Iterate all the markets
for market in tqdm(markets):
    
    # Ignore non-required symbols
    if not market.upper().endswith(QUOTE_CURRENCY):
        continue
        
    # Download Data
    tqdm.write(f'Downloading {market}...')
    data = []
    
    epoch_start = int(datetime.fromisoformat(START_DATE).timestamp() * 1000)    
    counter = 0
    while epoch_start < END_DATE and counter <= 5:
        
        raw_data = exchange.fetch_ohlcv(market, TIME_FRAME, epoch_start)
        tmp_df = pd.DataFrame(raw_data)

        if (len(tmp_df.index) == 0):
            epoch_start = epoch_start + (86400 * 1000)
            counter += 1
            continue
    
        tmp_df.index = [datetime.utcfromtimestamp(x // 1000) for x in tmp_df[0]]
        tmp_df.index.name = 'date'
        tmp_df = tmp_df[range(1, 6)]
        tmp_df.columns = ['open','high','low','close','volume']

        data.append(tmp_df)
        epoch_start = int(tmp_df.index.max().timestamp() * 1000 + 1000)

        counter = 0
        sleep(0.1)
        
    # If there is new data, concatenate all the downloaded chunks and save them to a file
    if len(data) > 0:
        df = pd.concat(data)
        df = df.sort_index()
        df.to_csv(f'{RESULT_DIR}/{market.replace("/","")}.csv.gz', compression='gzip')
