import os
import math
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import confusion_matrix, classification_report

# Constants
BARS=15
RESULT_DIR='../result'
DATE_SPLIT='2019-06-01'

# Get the data and generate the train, validation, and test datasets
def get_datasets():

    train_val = pd.read_csv('../data/spy.2008.2021.csv.gz', compression='gzip')
    train_val = train_val[['date','open','close']]
    train_val['date'] = pd.to_datetime(train_val['date'])

    test = pd.read_csv('../data/spy.csv.gz', compression='gzip')
    test = test[['date','open','close']]
    test['date'] = pd.to_datetime(test['date'])

    train = train_val[train_val['date'] <= DATE_SPLIT]
    validation = train_val[(train_val['date'] > DATE_SPLIT) & (train_val['date'] < test['date'].min())]

    return train, validation, test

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

    print(len(feature_result), len(targets))
    return np.array(feature_result), np.array(targets['trend'].values)

# Show the result of the operation
def show_result(target, pred, ds_type='TEST'):

    target = np.array(target)
    pred = np.array(pred)

    print(f' RESULT {ds_type.upper()} '.center(56, '*'))

    print('* Confusion Matrix (Top: Predicted - Left: Real)')
    print(confusion_matrix(y_true=target, y_pred=pred))

    print('* Classification Report')
    print(classification_report(target, pred))

# Save the results
def save_result(target, pred, name):

    os.makedirs(RESULT_DIR, exist_ok=True)
    df = pd.DataFrame.from_dict({'pred': pred, 'target': target})
    df.to_csv(f'{RESULT_DIR}/{name}', index=False)