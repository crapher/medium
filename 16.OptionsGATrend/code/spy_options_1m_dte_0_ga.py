from tqdm import tqdm
from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pygad

# Configuration
pd.set_option('display.float_format', lambda x: '%.4f' % x)

# Constants
BARS = 15        # Range: 0 - 30
STOP_LOSS = 0.7  # Range: 0 - 1 (0 -> 0% | 1 -> 100%)
POO = 0.01       # Range: 0 - 1 (0 -> 0% | 1 -> 100%)

OPTIONS_FILE='../data/spy_dte_0.csv.gz'
TRAIN_FILE='../data/spy.2008.2021.csv.gz'

FEES_PER_CONTRACT = 0.6
CASH = 1000

DATE_SPLIT = '2019-06-01'

SOLUTIONS = 30
GENERATIONS = 50

### Helper Methods ###

# Process Dataframe and return features and targets
def get_features_targets(df, scale_obs=True):

    feature_result = []
    dates = []

    # Remove duplicated dates
    df = df.groupby(by='date').mean().reset_index()

    # Get Features based on BARS configuration
    features = df[((df['date'].dt.hour == 9) & (df['date'].dt.minute >= 30)) &
                   (df['date'].dt.hour == 9) & (df['date'].dt.minute < 30 + BARS)]
    features = features.groupby(features['date'].dt.date)

    for dt, feature in features:

        if len(feature) != BARS:
            feature = feature.set_index('date')
            feature = feature.resample('1T').asfreq().reindex(pd.date_range(str(dt) + ' 09:30:00', str(dt) + f' 09:{30+BARS-1}:00', freq='1T'))
            feature = feature.reset_index()
            feature['close'] = feature['close'].fillna(method='ffill')
            feature['open'] = feature['open'].fillna(feature['close'])
            feature = feature.dropna()

        if len(feature) == BARS:
            feature = feature['close'].values

            if scale_obs:
                feature -= np.min(feature)
                feature /= np.max(np.abs(feature))
                feature = np.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)

            feature_result.append(feature)
            dates.append(dt)

    # Get Targets Trend based on first and last value / day (0: DOWN - 1: UP)
    targets = df.set_index('date')
    targets = targets.resample('1D').agg({'open':'first', 'close':'last'})
    targets = targets.loc[dates].reset_index().sort_values(by='date')
    targets['trend'] = np.where(targets['open'] < targets['close'], 1, 0)

    return np.array(feature_result), np.array(targets['trend'].values)

# Return the predicted values
def get_predicted_values(solution, features):

    pred_y = np.clip(np.dot(features, solution), 0, 1)
    pred_y = np.where(pred_y > 0.5, 1, 0)
    return pred_y

# Define fitness function to be used by the PyGAD instance
def fitness_func(self, solution, sol_idx):

    global train_x, train_y

    pred_y = get_predicted_values(solution, train_x)
    result = f1_score(train_y, pred_y, average='binary', pos_label=1) + \
             f1_score(train_y, pred_y, average='binary', pos_label=0)

    return result
    
### Read Files & Prepare Datasets ###
df_base = pd.read_csv(OPTIONS_FILE, header=0)
df_base['date'] = pd.to_datetime(df_base['date'])

train = pd.read_csv(TRAIN_FILE, header=0)
train['date'] = pd.to_datetime(train['date'])
train = train[train['date'] <= DATE_SPLIT]

test = df_base[['date', 'open_underlying', 'close_underlying']]
test.columns = ['date', 'open', 'close']
test = test.drop_duplicates().reset_index(drop=True)

### Prepare model to predict trend ###

# Get Features and Targets
train_x, train_y = get_features_targets(train)
test_x, _ = get_features_targets(test)

# Train Model
with tqdm(total=GENERATIONS) as pbar:

    # Create Genetic Algorithm
    ga_instance = pygad.GA(num_generations=GENERATIONS,
                           num_parents_mating=5,
                           fitness_func=fitness_func,
                           sol_per_pop=SOLUTIONS,
                           num_genes=BARS,
                           gene_space={'low': -1, 'high': 1},
                           random_seed=42,
                           on_generation=lambda _: pbar.update(1),
                           )

    # Run the Genetic Algorithm
    ga_instance.run()

# Get the best solution
solution, _, _ = ga_instance.best_solution()

### Get the trend of each day to see which option we should buy ###

# Get first bar (To get the Option Open Price)
df_day_open = df_base[(df_base['date'].dt.hour == 9) & (df_base['date'].dt.minute == 30)]

# Get *BARS* bar (To get Underlying Close Price)
df = df_base[(df_base['date'].dt.hour == 9) & (df_base['date'].dt.minute == 30 + BARS - 1)]

# Add the Option Open Price
df = df.merge(df_day_open[['expire_date','strike','kind','open']],
              how='left',
              left_on=['expire_date','strike','kind'],
              right_on=['expire_date','strike','kind'],
              suffixes=('','_dayopen'))

# Keep the first open value for each strike
df = df.rename(columns={'open_dayopen': 'option_open'})

# Predict Trend and add to test df
test['date'] = test['date'].dt.date.astype(str)
test = test[['date']].drop_duplicates().reset_index(drop=True)
test['trend'] = get_predicted_values(solution, test_x)
test['trend'] = np.where(test['trend'] == 0, -1, test['trend']) 

# Add Trend to df
df = df.merge(test,
              how='left',
              left_on=['expire_date'],
              right_on=['date'],
              suffixes=['','_ga'])

# Remove all previous merged values for trend calculation and rows with NaN values
df = df.loc[:,~df.columns.str.endswith('_ga')]
df = df.dropna()

### Get the closest ITM option ###

# Filter all puts when trend is going down and calls when trend is going up
df = df[((df['kind'] == 'P') & (df['trend'] == -1)) |
        ((df['kind'] == 'C') & (df['trend'] == 1))]

# Calculate Strike distance from Underlying price
df['distance'] = df['trend'] * (df['close_underlying'] - df['strike'])

# Remove OTM & ATM Options
df = df[df['distance'] > 0]

# Get the closest ITM options
idx = df.groupby(['expire_date'])['distance'].transform(min) == df['distance']
df = df[idx]

# Remove distance column
df = df.drop('distance', axis=1)

### Calculate close points ###

# Get trade bars
df_trade = df_base[((df_base['date'].dt.hour == 9) & (df_base['date'].dt.minute > 30 + BARS - 1)) |
                    (df_base['date'].dt.hour >= 10)]

# Get Option Open and Close Points
df = df_trade.merge(df[['expire_date','kind','strike','option_open']],
                    how='right',
                    left_on=['expire_date','kind','strike'],
                    right_on=['expire_date','kind','strike'])

df.loc[:,'open_point'] = np.where((df['open'] >= df['option_open'] * (1 + POO)) &
                                  ((df['open'].shift() < df['option_open'].shift() * (1 + POO)) |
                                   (df['expire_date'] != df['expire_date'].shift())), 1, 0)

df.loc[:,'stop_loss'] = df['option_open'] * STOP_LOSS
df.loc[:,'last_date'] = df.groupby(['expire_date','kind','strike'])['date'].transform('last')

df.loc[:,'close_point'] = np.where(((df['close'] <= df['stop_loss']) &
                                    ((df['close'].shift() > df['stop_loss'].shift()) | (df['expire_date'] != df['expire_date'].shift()))) |
                                   (df['last_date'] == df['date']), 1, 0)

df_tmp = df[(df['open_point'] - df['close_point']) == 0]
df = df[(df['open_point'] - df['close_point']) != 0]
df.loc[:,'open_point'] = np.where((df['open_point'] - df['close_point']) == (df['open_point'].shift(-1) - df['close_point'].shift(-1)), 0, df['open_point'])

df = pd.concat([df, df_tmp])
df = df.sort_values(by=['date','expire_date','kind','strike'])

# Get Open Price, Close Price, Open Date, and Close Date
df = df[(df['open_point'] != 0) | (df['close_point'] != 0)]

df['open_price'] = np.where(df['open_point'] == 1, df['open'], np.NaN)
df['close_price'] = np.where(df['close_point'] == 1, df['close'], np.NaN)
df['close_price'] = df['close_price'].fillna(method='bfill', limit=1)

df['close_date'] = np.where(df['open_point'] - df['close_point'] == 0, df['date'], df['date'].shift(-1))
df = df.rename(columns={'date':'open_date'})

# Clean all Rows with NaN values (This is going to remove all invalid closes)
df = df.dropna()

# Clean all the unneeded columns
df = df.drop(['last_date','open_point','close_point','open','close'], axis=1)
df = df.loc[:,~df.columns.str.endswith('_underlying')]

# Save the trigger of the closing
df.loc[:,'trigger'] = np.where(df['close_price'] <= df['stop_loss'], 'SL', 'EXPIRED')

### Generate result ###

# Calculate the variables required in the result
df['contracts'] = (CASH // (100 * df['open_price'])).astype(int)
df['fees'] = np.where(df['trigger'] == 'EXPIRED', FEES_PER_CONTRACT, 2 * FEES_PER_CONTRACT) * df['contracts']
df['gross_result'] = df['contracts'] * 100 * (df['close_price'] - df['open_price'])
df['net_result'] = df['gross_result'] - df['fees']

sl = len(df[df["trigger"] == "SL"])
exp = len(df[df["trigger"] != "SL"])
total = len(df)

# Configuration
print(f' CONFIGURATION '.center(70, '*'))
print(f'* Bars: {BARS}')
print(f'* Stop Loss: {STOP_LOSS * 100:.0f}%')
print(f'* Percentage over price: {POO * 100:.0f}%')

# Show the Total Result
print(f' SUMMARIZED RESULT '.center(70, '*'))
print(f'* Trading Days: {len(df["expire_date"].unique())}')
print(f'* Operations: {len(df)} - Stop Loss: {sl} ({100 * sl / total:.2f}%) - Expired: {exp} ({100 * exp / total:.2f}%)')
print(f'* Gross PnL: $ {df["gross_result"].sum():.2f}')
print(f'* Net PnL: $ {df["net_result"].sum():.2f}')

# Show The Monthly Result
print(f' MONTHLY DETAIL RESULT '.center(70, '*'))
df_monthly = df[['expire_date','gross_result','net_result']]
df_monthly['year_month'] = df_monthly['expire_date'].str[0:7]
df_monthly = df_monthly.groupby(['year_month'])[['gross_result','net_result']].sum()
print(df_monthly)
