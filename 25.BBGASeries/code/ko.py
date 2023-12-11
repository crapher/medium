import numpy as np
import pandas as pd
import pandas_ta as ta
import pygad

from tqdm import tqdm

### User configuration ###
TICKER = 'KO'
CASH = 10_000                       # Cash available for operations

BB_SMA = 20                         # Bollinger bands SMA
BB_STD = 2.0                        # Bollinger bands standard deviation
BB_MAX_BANDWIDTH = 5                # Bollinger bands maximum volatility allowed

DAYS_FOR_TESTING = 365 * 1.5        # Days used for testing
WINDOW_REWARD = '3M'                # Window used to calculate the reward of a solution
WINDOW_MIN_OPERATIONS = 21 * 3      # Minimum operations quantity required to calculate the reward

GENERATIONS = 50                    # Iterations count used by the genetic algorithm
SOLUTIONS = 20                      # Solutions / iteration calculated by the genetic algorithm

### Constants ###
FILENAME = f'../data/{TICKER.lower()}.csv.gz'
TIMEFRAMES = ['5T','15T','1H']

### Data format & preparation ###
BB_SMA = int(BB_SMA)
BB_STD = round(BB_STD, 2)
BB_UPPER = f'BBU_{int(BB_SMA)}_{BB_STD}'
BB_LOWER = f'BBL_{int(BB_SMA)}_{BB_STD}'
BB_VOLATILITY = f'BBB_{int(BB_SMA)}_{BB_STD}'

DAYS_FOR_TESTING = int(DAYS_FOR_TESTING)
WINDOW_MIN_OPERATIONS = int(WINDOW_MIN_OPERATIONS)

### Output preparation ###
np.set_printoptions(suppress=True)
pd.options.mode.chained_assignment = None

### Data functions ###
def get_data(ticker, timeframe):

    # Read data from file
    df = pd.read_csv(FILENAME)
    df['date'] = pd.to_datetime(df['date'])

    df = df.set_index('date').resample(timeframe).agg({'close':'last'}).dropna().reset_index()

    # Calculate bollinger bands based on configuration
    df.ta.bbands(close=df['close'], length=BB_SMA, std=BB_STD, append=True)
    df = df.dropna()

    # Calculate limits (lower: 25% - upper: 75%), close percentage, and volatility
    df['high_limit'] = df[BB_UPPER] + (df[BB_UPPER] - df[BB_LOWER]) / 2
    df['low_limit'] = df[BB_LOWER] - (df[BB_UPPER] - df[BB_LOWER]) / 2
    df['close_percentage'] = np.clip((df['close'] - df['low_limit']) / (df['high_limit'] - df['low_limit']), 0, 1)
    df['volatility'] = np.clip(df[BB_VOLATILITY] / (100 / BB_MAX_BANDWIDTH), 0, 1)

    # Remove all the bollinger bands fields that won't be needed from now on
    df = df.loc[:,~df.columns.str.startswith('BB')]

    # Split the data in train and test
    train = df[df['date'].dt.date <= (df['date'].dt.date.max() - pd.Timedelta(DAYS_FOR_TESTING, 'D'))]
    test = df[df['date'] > train['date'].max()]

    return train, test

def get_result(df, min_volatility, max_buy_perc, min_sell_perc):

    # Generate a copy to avoid changing the original data
    df = df.copy().reset_index(drop=True)        

    # Buy signal
    df['signal'] = np.where((df['volatility'] > min_volatility) & (df['close_percentage'] < max_buy_perc), 1, 0)

    # Sell signal
    df['signal'] = np.where((df['close_percentage'] > min_sell_perc), -1, df['signal'])

    # Remove all rows without operations, rows with the same consecutive operation, first row selling, and last row buying
    result = df[df['signal'] != 0]
    result = result[result['signal'] != result['signal'].shift()]
    if (len(result) > 0) and (result.iat[0, -1] == -1): result = result.iloc[1:]
    if (len(result) > 0) and (result.iat[-1, -1] == 1): result = result.iloc[:-1]

    # Calculate pnl, wins, losses, and reward / operation
    result['pnl'] = np.where(result['signal'] == -1, (result['close'] - result['close'].shift()) * (CASH // result['close'].shift()), 0)
    result['wins'] = np.where(result['pnl'] > 0, 1, 0)
    result['losses'] = np.where(result['pnl'] < 0, 1, 0)

    # Remove bars without operations
    result = result[result['signal'] == -1]
        
    # Remove the signal column and return the dataset
    return result.drop('signal', axis=1)

def calculate_reward(df):

    # Generate window to calculate reward average
    df_reward = df.set_index('date').resample(WINDOW_REWARD).agg(
        {'close':'last','wins':'sum','losses':'sum','pnl':'sum'}).reset_index()

    # Generate reward
    wins = df_reward['wins'].mean() if len(df_reward) > 0 else 0
    losses = df_reward['losses'].mean() if len(df_reward) > 0 else 0
    reward = df_reward['pnl'].mean() if (WINDOW_MIN_OPERATIONS < (wins + losses)) else -WINDOW_MIN_OPERATIONS + (wins + losses)

    return reward

def show_result(df, name, show_monthly):

    # Calculate required values
    reward = calculate_reward(df)
    pnl = df['pnl'].sum()
    wins = df['wins'].sum() if len(df) > 0 else 0
    losses = df['losses'].sum() if len(df) > 0 else 0
    win_rate = (100 * (wins / (wins + losses)) if wins + losses > 0 else 0)
    max_profit = df['pnl'].max()
    min_drawdown = df['pnl'].min()
    avg_pnl = df['pnl'].mean()

    # Show the summarized result
    print(f' SUMMARIZED RESULT - {name} '.center(60, '*'))
    print(f'* Reward              : {reward:.2f}')
    print(f'* Profit / Loss       : {pnl:.2f}')
    print(f'* Wins / Losses       : {wins:.0f} / {losses:.0f} ({win_rate:.2f}%)')
    print(f'* Max Profit          : {max_profit:.2f}')
    print(f'* Max Drawdown        : {min_drawdown:.2f}')
    print(f'* Profit / Loss (Avg) : {avg_pnl:.2f}')

    # Show the monthly result
    if show_monthly:
        print(f' MONTHLY DETAIL RESULT '.center(60, '*'))
        df_monthly = df.set_index('date').resample('1M').agg(
            {'wins':'sum','losses':'sum','pnl':'sum'}).reset_index()
        df_monthly = df_monthly[['date','pnl','wins','losses']]
        df_monthly['year_month'] = df_monthly['date'].dt.strftime('%Y-%m')
        df_monthly = df_monthly.drop('date', axis=1)
        df_monthly = df_monthly.groupby(['year_month']).sum()
        df_monthly['win_rate'] = round(100 * df_monthly['wins'] / (df_monthly['wins'] + df_monthly['losses']), 2)

        print(df_monthly)

### Genetic algorithm functions ###
def fitness_func(self, solution, sol_idx):

    # Get reward from train data
    result = get_result(train, solution[0], solution[1], solution[2])

    # Return the solution reward
    return calculate_reward(result)

def get_best_solution():

    with tqdm(total=GENERATIONS) as pbar:

        # Create genetic algorithm
        ga_instance = pygad.GA(num_generations=GENERATIONS,
                               num_parents_mating=5,
                               fitness_func=fitness_func,
                               sol_per_pop=SOLUTIONS,
                               num_genes=3,
                               gene_space=[
                                {'low': 0, 'high': 1, 'step': 0.0001},
                                {'low': 0, 'high': 1, 'step': 0.0001},
                                {'low': 0, 'high': 1, 'step': 0.0001}],
                               parent_selection_type='sss',
                               crossover_type='single_point',
                               mutation_type='random',
                               mutation_num_genes=1,
                               keep_parents=-1,
                               random_seed=42,
                               on_generation=lambda _: pbar.update(1),
                               )

        # Run the genetic algorithm
        ga_instance.run()

    # Return the best solution
    return ga_instance.best_solution()[0]

### Main function ###
def main(ticker):

    global train

    for timeframe in TIMEFRAMES:

        # Get Train and Test data for timeframe
        train, test = get_data(ticker, timeframe)

        # Process timeframe
        print(''.center(60, '*'))
        print(f' PROCESSING {ticker.upper()} - TIMEFRAME {timeframe} '.center(60, '*'))
        print(''.center(60, '*'))

        solution = get_best_solution()

        print(f' Best Solution Parameters '.center(60, '*'))
        print(f'Min Volatility   : {solution[0]:6.4f}')
        print(f'Max Perc to Buy  : {solution[1]:6.4f}')
        print(f'Min Perc to Sell : {solution[2]:6.4f}')

        # Show the train result
        result = get_result(train, solution[0], solution[1], solution[2])
        show_result(result, f'TRAIN ({train["date"].min().date()} - {train["date"].max().date()})', False)

        # Show the test result
        result = get_result(test, solution[0], solution[1], solution[2])
        show_result(result, f'TEST ({test["date"].min().date()} - {test["date"].max().date()})', True)

        print('')

if __name__ == '__main__':
    main(TICKER)