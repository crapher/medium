import pandas as pd

BUNDLE_URL = 'https://www.cftc.gov/files/dea/history/fin_fut_txt_2006_2016.zip'
YEAR_URL = 'https://www.cftc.gov/files/dea/history/fut_fin_txt_{}.zip'

def get_dataframe(url):
    hdr = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11'}

    df = pd.read_csv(url, compression='zip', storage_options=hdr, low_memory=False)
    df = df[['Market_and_Exchange_Names',
         'Report_Date_as_YYYY-MM-DD',
         'Pct_of_OI_Dealer_Long_All',
         'Pct_of_OI_Dealer_Short_All',
         'Pct_of_OI_Lev_Money_Long_All',
         'Pct_of_OI_Lev_Money_Short_All']]

    df['Report_Date_as_YYYY-MM-DD'] = pd.to_datetime(df['Report_Date_as_YYYY-MM-DD'])
    return df

df = get_dataframe(BUNDLE_URL)
df = df[df['Report_Date_as_YYYY-MM-DD'] < '2016-01-01']

for year in range(2016, 2024):
    tmp_df = get_dataframe(YEAR_URL.format(year))
    df = pd.concat([df, tmp_df])

df = df.sort_values(['Market_and_Exchange_Names','Report_Date_as_YYYY-MM-DD']).reset_index(drop=True)
df = df.drop_duplicates()
df.to_csv('../data/market_sentiment_data.csv.gz', index=False)