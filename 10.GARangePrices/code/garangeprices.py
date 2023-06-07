import numpy as np
import pandas as pd
import pandas_ta as ta
import pygad

from tqdm import tqdm

# Constants
SOLUTIONS = 20
GENERATIONS = 50
DAYS = 7
TIMEFRAMES = ['5T','15T','1H','1D']
LEN = {'5T': int(6.5 * DAYS * 12), '15T': int(6.5 * DAYS * 4), '1H': int(6.5 * DAYS), '1D': DAYS}

# Configuration
np.set_printoptions(suppress=True)
pd.options.mode.chained_assignment = None

# Loading data, and split in train and test datasets
def get_data(timeframe, length):

    # Read the data
    df = pd.read_csv(f'OIH_{timeframe}.csv.gz', compression='gzip')

    # Get close in LEN bars in the future
    df['close_future'] = df['close'].shift(-length)

    # Get High/Low in LEN bars in the future
    df['high_future'] = df['high'].shift(-length).rolling(length).max()
    df['low_future'] = df['low'].shift(-length).rolling(length).min()
    
    # Calculate Moving Volatility & Upper/Lower limits
    df['change'] = np.log(df['close'] / df['close'].shift())
    df['volatility'] = df['change'].rolling(length).agg(lambda c: c.std() * length ** .5)
    
    df['upper_limit'] = df['close'] * (1 + df['volatility'])
    df['lower_limit'] = df['close'] * (1 - df['volatility'])
    
    # Calculate Trend
    df['ema200'] = ta.ema(df['close'], length=200)
    df['ema50'] = ta.ema(df['close'], length=50)
    
    df['trend_up'] = df['ema200'] < df['ema50']
    
    # Clean all NaN values
    df = df.dropna()
    
    # Calculate the close percentage relative to limits 
    df['close_perc'] = np.clip((df['close_future'] - df['lower_limit']) / (df['upper_limit'] - df['lower_limit']), 0, 1)
    
    # Check values out of bounds
    df['out_of_bounds'] = ((df['high_future'] > df['upper_limit']) & (df['trend_up'] == True)) | ((df['low_future'] < df['lower_limit']) & (df['trend_up'] == False))
    
    # Split Train and Test datasets
    train = df[df['date'] < '2022-01-01']
    test = df[df['date'] >= '2022-01-01']

    return train, test

# Define fitness function to be used by the PyGAD instance
def fitness_func(self, solution, sol_idx):

    # Get Reward from train data
    total_reward, _, _, _ = get_result(train, solution[0], solution[1], solution[2])

    # Return the solution reward
    return total_reward

# Define a reward function
def get_result(df, min_volatility, bottom_perc, top_perc):
    
    # Get Total data Len
    total_len = len(df)
    
    # Filter data
    df = df[(df['close_perc'] > bottom_perc) & (df['close_perc'] < top_perc) & (df['volatility'] > min_volatility)]
    after_filter_len = len(df)
    
    # Get values under/over limit
    out_of_bounds = df['out_of_bounds'].sum()
    
    # Calculate Reward
    if after_filter_len > 0:
        in_bounds = after_filter_len - out_of_bounds
        percentage = in_bounds / after_filter_len
        total_reward = percentage * in_bounds - ((1 - percentage) * out_of_bounds)
    else:
        total_reward = -1

    return total_reward, total_len, after_filter_len, out_of_bounds

for timeframe in TIMEFRAMES:

    # Get Train and Test data for timeframe
    train, test = get_data(timeframe, LEN[timeframe])

    # Process timeframe
    print("".center(60, "*"))
    print(f' PROCESSING TIMEFRAME {timeframe} '.center(60, '*'))
    print("".center(60, "*"))

    with tqdm(total=GENERATIONS) as pbar:

        # Create Genetic Algorithm
        ga_instance = pygad.GA(num_generations=GENERATIONS,
                               num_parents_mating=5,
                               fitness_func=fitness_func,
                               sol_per_pop=SOLUTIONS,
                               num_genes=3,
                               gene_space=[{'low': 0, 'high':0.05}, {'low': 0, 'high':1}, {'low': 0, 'high':1}],
                               parent_selection_type="sss",
                               crossover_type="single_point",
                               mutation_type="random",
                               mutation_num_genes=1,
                               keep_parents=-1,
                               on_generation=lambda _: pbar.update(1),
                               )

        # Run the Genetic Algorithm
        ga_instance.run()

    # Show details of the best solution.
    solution, solution_fitness, _ = ga_instance.best_solution()

    print(f' Best Solution Parameters '.center(60, '*'))
    print(f"* Min Volatility       : {solution[0]:.4f}")
    print(f"* Bottom Perc.         : {solution[1]:.4f}")
    print(f"* Top Perc.            : {solution[2]:.4f}")

    # Get Reward from train data
    total_reward, total_len, after_filter_len, out_of_bounds = get_result(train, solution[0], solution[1], solution[2])

    print(f' Result for timeframe {timeframe} (TRAIN) '.center(60, '*'))
    print(f"* Total Records        : {total_len}")
    print(f"* Records after filter : {after_filter_len}")
    print(f"* Out Of Bounds        : {out_of_bounds} ({100 * (out_of_bounds / after_filter_len):.1f}%)")
    print(f"* Inside Bounds        : {after_filter_len - out_of_bounds} ({100*((after_filter_len - out_of_bounds) / after_filter_len):.1f}%)")

    # Get Reward from test data
    total_reward, total_len, after_filter_len, out_of_bounds = get_result(test, solution[0], solution[1], solution[2])

    # Show the final result
    print(f' Result for timeframe {timeframe} (TEST) '.center(60, '*'))
    print(f"* Total Records        : {total_len}")
    print(f"* Records after filter : {after_filter_len}")
    print(f"* Out Of Bounds        : {out_of_bounds} ({100 * (out_of_bounds / after_filter_len):.1f}%)")
    print(f"* Inside Bounds        : {after_filter_len - out_of_bounds} ({100*((after_filter_len - out_of_bounds) / after_filter_len):.1f}%)")

    print("")
