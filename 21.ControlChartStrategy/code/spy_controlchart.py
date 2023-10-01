import os
import numpy as np
import pandas as pd

from scipy import signal

# Constants
FILENAME = '../data/spy.csv.gz'
DEFAULT_WINDOW = 10
CASH = 10_000

# Configuration
np.set_printoptions(suppress=True)
pd.options.mode.chained_assignment = None

# Loading dataset
def get_data(filename):

    df = pd.read_csv(filename, compression='gzip')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').resample('5T').agg('last')
    df = df.dropna()
    df['feature'] = signal.detrend(df['close'])

    return df.reset_index(drop=True)

# Show result based on the selected rule
def show_result(df, signal_field):

    # Remove all rows without operations, rows with the same consecutive operation, first row selling, and last row buying
    ops = df[df[signal_field] != 0]
    ops = ops[ops[signal_field] != ops[signal_field].shift()]
    if (len(ops) > 0) and (ops.iat[0, -1] == -1): ops = ops.iloc[1:]
    if (len(ops) > 0) and (ops.iat[-1, -1] == 1): ops = ops.iloc[:-1]

    # Calculate P&L / operation
    ops['pnl'] = np.where(ops[signal_field] == -1, (ops['close'] - ops['close'].shift()) * (CASH // ops['close'].shift()), 0)

    # Calculate total P&L, wins, and losses
    pnl = ops['pnl'].sum()
    wins = len(ops[ops['pnl'] > 0])
    losses = len(ops[ops['pnl'] < 0])

    # Show Result
    print(f' Result ({signal_field}) '.center(60, '*'))
    print(f"* Profit / Loss  : {pnl:.2f}")
    print(f"* Wins / Losses  : {wins} / {losses}")
    print(f"* Win Rate       : {(100 * (wins/(wins + losses)) if wins + losses > 0 else 0):.2f}%")

# Rules definition
def apply_rule_1(df, window = DEFAULT_WINDOW):

    # One point beyond the 3 stdev control limit

    df['sma'] = df['feature'].rolling(window=window).mean()
    df['3std'] = 3 * df['feature'].rolling(window=window).std()

    df['rule1'] = np.where(df['feature'] < df['sma'] - df['3std'], 1, 0)
    df['rule1'] = np.where(df['feature'] > df['sma'] - df['3std'], -1, df['rule1'])

    return df.drop(['sma','3std'], axis=1)

def apply_rule_2(df, window = DEFAULT_WINDOW):

    # Eight or more points on one side of the centerline without crossing

    df['sma'] = df['feature'].rolling(window=window).mean()

    for side in ['upper', 'lower']:
        df['count_' + side] = (df['feature'] > df['sma']) if side == 'upper' else (df['feature'] < df['sma'])
        df['count_' + side] = df['count_' + side].astype(int)
        df['count_' + side] = df['count_' + side].rolling(window=8).sum()

    df['rule2'] = np.where(df['count_upper'] >= 8, 1, 0)
    df['rule2'] = np.where(df['count_lower'] >= 8, -1, df['rule2'])

    return df.drop(['sma','count_upper','count_lower'], axis=1)

def apply_rule_3(df, window = DEFAULT_WINDOW):

    # Four out of five points over 1 stdev or under -1 stdev

    df['sma'] = df['feature'].rolling(window=window).mean()
    df['1std'] = df['feature'].rolling(window=window).std()

    df['rule3'] = np.where((df['feature'] < df['sma'] - df['1std']).rolling(window=5).sum() >= 4, 1, 0)
    df['rule3'] = np.where((df['feature'] > df['sma'] + df['1std']).rolling(window=5).sum() >= 4, -1, df['rule3'])

    return df.drop(['sma','1std'], axis=1)

def apply_rule_4(df):

    # Six points or more in a row steadily increasing or decreasing

    df['rule4'] = np.where((df['feature'] < df['feature'].shift(1)) &
                           (df['feature'].shift(1) < df['feature'].shift(2)) &
                           (df['feature'].shift(2) < df['feature'].shift(3)) &
                           (df['feature'].shift(3) < df['feature'].shift(4)) &
                           (df['feature'].shift(4) < df['feature'].shift(5)), 1, 0)

    df['rule4'] = np.where((df['feature'] > df['feature'].shift(1)) &
                           (df['feature'].shift(1) > df['feature'].shift(2)) &
                           (df['feature'].shift(2) > df['feature'].shift(3)) &
                           (df['feature'].shift(3) > df['feature'].shift(4)) &
                           (df['feature'].shift(4) > df['feature'].shift(5)), -1, df['rule4'])

    return df

def apply_rule_5(df, window = DEFAULT_WINDOW):

    # Two out of three points over 2 stdev or under -2 stdev

    df['sma'] = df['feature'].rolling(window=window).mean()
    df['2std'] = 2 * df['feature'].rolling(window=window).std()

    df['rule5'] = np.where((df['feature'] < df['sma'] - df['2std']).rolling(window=3).sum() >= 2, 1, 0)
    df['rule5'] = np.where((df['feature'] > df['sma'] + df['2std']).rolling(window=3).sum() >= 2, -1, df['rule5'])

    return df.drop(['sma','2std'], axis=1)

def apply_rule_6(df, window = DEFAULT_WINDOW):

    # 14 points in a row alternating up and down

    df['sma'] = df['feature'].rolling(window=window).mean()
    df['1std'] = df['feature'].rolling(window=window).std()
    df['2std'] = 2 * df['1std']

    # Determine the zones for each row
    df['zone'] = None
    df.loc[df['feature'] > df['sma'], 'zone'] = '+C'
    df.loc[df['feature'] > df['sma'] + df['1std'], 'zone'] = '+B'
    df.loc[df['feature'] > df['sma'] + df['2std'], 'zone'] = '+A'
    df.loc[df['feature'] < df['sma'], 'zone'] = '-C'
    df.loc[df['feature'] < df['sma'] - df['1std'], 'zone'] = '-B'
    df.loc[df['feature'] < df['sma'] - df['2std'], 'zone'] = '-A'

    df['rule6'] = np.where((df['zone'] != df['zone'].shift()).rolling(window=14).sum() >= 14, 1, -1)

    return df.drop(['sma','1std','2std','zone'], axis=1)

df = get_data(FILENAME)

df = apply_rule_1(df)
show_result(df, 'rule1')

df = apply_rule_2(df)
show_result(df, 'rule2')

df = apply_rule_3(df)
show_result(df, 'rule3')

df = apply_rule_4(df)
show_result(df, 'rule4')

df = apply_rule_5(df)
show_result(df, 'rule5')

df = apply_rule_6(df)
show_result(df, 'rule6')
