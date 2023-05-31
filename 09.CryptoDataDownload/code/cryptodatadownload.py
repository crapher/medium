import io
import requests as rq
import pandas as pd

# Constants
URL_FORMAT='https://www.cryptodatadownload.com/cdd/Gemini_ETHUSD_{}_minute.csv'

df = pd.DataFrame()

# Iterate the years we want to download
for year in range(2016, 2024):

    url = URL_FORMAT.format(year)

    # Download the information
    r = rq.get(url, verify=False)
    r.raise_for_status()

    # Prepare and generate the temporary dataframe with the downloaded information
    tmp_df = pd.read_csv(io.StringIO(r.text), header=1)
    tmp_df = tmp_df.drop(['unix','symbol'], axis=1)
    
    # If there are 2 Volume columns, remove the last one (USD)
    if 'Volume' in tmp_df.columns[-2] and 'Volume' in tmp_df.columns[-1]:
        tmp_df = tmp_df.iloc[: , :-1]

    tmp_df['date'] = pd.to_datetime(tmp_df['date'])
    tmp_df = tmp_df.sort_values('date').reset_index(drop=True).set_index('date')
    
    # Add the Temporary dataframe to the main dataframe
    df = pd.concat([df, tmp_df])

# Rename the columns to lowercase and replace any space in the column names with underscores
df.columns = ['open','high','low','close','volume']

# Set the index name and reset the index
df.index.name = 'date'

# Convert the data to different timeframes & save them for future uses
AGGREGATION = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
TIMEFRAMES = ['1T', '5T', '15T', '1H', '1D']

for timeframe in TIMEFRAMES:
    print(f'Converting & Saving {timeframe} Data...')
    df = df.resample(timeframe).agg(AGGREGATION).dropna()
    df.to_csv(f'OIH_{timeframe}.csv.gz', compression='gzip')

