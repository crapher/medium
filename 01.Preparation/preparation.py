### This file is the complete code of this Medium article
### https://medium.com/@diegodegese/backtesting-stock-trading-strategies-using-python-ea191a15970c
import pandas as pd

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

# Download data
print(f'Downloading OIH_adjusted.txt...')
urlretrieve('http://api.kibot.com/?action=history&symbol=OIH&interval=1&unadjusted=0&bp=1&user=guest', 'OIH_adjusted.txt')

# Read data and assign names to the columns
df = pd.read_csv('OIH_adjusted.txt')
df.columns = ['date','time','open','high','low','close','volume']

# Combine date and time in the date column
df['date'] = df['date'] + ' ' + df['time']
df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y %H:%M')
df = df[['date','open','high','low','close','volume']]

# Sort by date and assign the date as index
df = df.sort_values('date').reset_index(drop=True).set_index('date')

# Convert the data to different timeframes & save them for future uses
AGGREGATION = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
TIMEFRAMES = ['1T', '5T', '15T', '1H', '1D']

for timeframe in TIMEFRAMES:
    print(f'Converting & Saving {timeframe} Data...')
    df = df.resample(timeframe).agg(AGGREGATION).dropna()
    df.to_csv(f'OIH_{timeframe}.csv.gz', compression='gzip')
