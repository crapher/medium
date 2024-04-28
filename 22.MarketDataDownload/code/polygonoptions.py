import os
import pandas as pd

from tqdm import tqdm
from datetime import datetime, timedelta
from polygon import RESTClient, exceptions

API_KEY = 'YOUR_API_KEY'

UNDERLYING_TICKER = 'SPY'

FILEPATH = f'./{UNDERLYING_TICKER.lower()}'
DATAPATH = f'{FILEPATH}/rawdata'
INDEX_FILE = f'{FILEPATH}/index.csv.gz'

EXPIRATION_DATE_MIN = (datetime.now() - timedelta(days=2 * 365)).strftime("%Y-%m-%d")
MAX_DAYS_TO_EXPIRATION = 7

os.makedirs(FILEPATH, exist_ok=True)

client = RESTClient(api_key=API_KEY)

contracts = []

for contract in client.list_options_contracts(
    underlying_ticker=UNDERLYING_TICKER, 
    expiration_date_gte=EXPIRATION_DATE_MIN,
    expired=True,
    limit=1000):
    contracts.append(contract)

df_index = pd.DataFrame.from_dict(contracts)
df_index.to_csv(INDEX_FILE, index=False, compression='gzip')
df_index['expiration_date'] = pd.to_datetime(df_index['expiration_date'])

os.makedirs(DATAPATH, exist_ok=True)

for contract in tqdm(df_index.to_dict('records')):
    
    expire_date = contract['expiration_date']
    day_folder = f'{DATAPATH}/{expire_date.strftime("%Y/%m/%d")}'
    file_name = f'{day_folder}/{contract["ticker"]}.csv.gz'.replace('O:','')
    
    start_date = expire_date - timedelta(days=MAX_DAYS_TO_EXPIRATION)
    end_date = expire_date + timedelta(days=1)

    try:
        aggs = client.get_aggs(
            contract['ticker'], 1, 'minute',
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
            )

        df_data = pd.DataFrame.from_dict(aggs)
        if df_data.empty:
            continue
        
        os.makedirs(day_folder, exist_ok=True)

        df_data['date'] = pd.to_datetime(df_data['timestamp'], unit='ms')
        df_data['strike'] = contract['strike_price']        
        df_data['expire_date'] = contract['expiration_date']
        df_data = df_data[['date','expire_date','strike','open','high','low','close','volume']]
        df_data.to_csv(file_name, index=False, compression='gzip')
        
    except:
        pass
