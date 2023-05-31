import numpy as np
import pandas as pd
import pandas_ta as ta
import pygad

from tqdm import tqdm

# Constants
SOLUTIONS = 20
GENERATIONS = 50
TIMEFRAMES = ['5T','15T','1H','1D']

# Configuration
np.set_printoptions(suppress=True)
pd.options.mode.chained_assignment = None

# Loading data, and split in train and test datasets
def get_data(timeframe):

    df = pd.read_csv(f'OIH_{timeframe}.csv.gz', compression='gzip')
    df.ta.bbands(close=df['close'], length=20, append=True)
    df = df.dropna()
    df['high_limit'] = df['BBU_20_2.0'] + (df['BBU_20_2.0'] - df['BBL_20_2.0']) / 2
    df['low_limit'] = df['BBL_20_2.0'] - (df['BBU_20_2.0'] - df['BBL_20_2.0']) / 2
    df['close_percentage'] = np.clip((df['close'] - df['low_limit']) / (df['high_limit'] - df['low_limit']), 0, 1)
    df['volatility'] = df['BBU_20_2.0'] / df['BBL_20_2.0'] - 1

    train = df[df['date'] < '2022-01-01']
    test = df[df['date'] >= '2022-01-01']

    return train, test

# Define fitness function to be used by the PyGAD instance
def fitness_func(self, solution, sol_idx):

    # Get Reward from train data
    total_reward, _, _ = get_result(train, solution[0], solution[1], solution[2])

    # Return the solution reward
    return total_reward

# Define a reward function
def get_result(df, min_volatility, max_buy_perc, min_sell_perc):

    # Generate a copy to avoid changing the original data
    df = df.copy().reset_index(drop=True)

    # Buy Signal
    df['signal'] = np.where((df['volatility'] > min_volatility) & (df['close_percentage'] < max_buy_perc), 1, 0)

    # Sell Signal
    df['signal'] = np.where((df['close_percentage'] > min_sell_perc), -1, df['signal'])

    # Remove all rows without operations, rows with the same consecutive operation, first row selling, and last row buying
    result = df[df['signal'] != 0]
    result = result[result['signal'] != result['signal'].shift()]
    if (len(result) > 0) and (result.iat[0, -1] == -1): result = result.iloc[1:]
    if (len(result) > 0) and (result.iat[-1, -1] == 1): result = result.iloc[:-1]

    # Calculate the reward / operation
    result['total_reward'] = np.where(result['signal'] == -1, result['close'] - result['close'].shift(), 0)

    # Generate the result
    total_reward = result['total_reward'].sum()
    wins = len(result[result['total_reward'] > 0])
    losses = len(result[result['total_reward'] < 0])
    
    return total_reward, wins, losses

for timeframe in TIMEFRAMES:

    # Get Train and Test data for timeframe
    train, test = get_data(timeframe)

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
                               gene_space=[{'low': 0, 'high':1}, {'low': 0, 'high':1}, {'low': 0, 'high':1}],
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

    #print(f"Fitness value of the best solution: {solution_fitness:10.2f}")
    print(f' Best Solution Parameters '.center(60, '*'))
    print(f"Min Volatility   : {solution[0]:6.4f}")
    print(f"Min Perc to Buy  : {solution[1]:6.4f}")
    print(f"Max Perc to Sell : {solution[2]:6.4f}")

    # Get Reward from train data
    profit, wins, losses = get_result(train, solution[0], solution[1], solution[2])

    print(f' Result for timeframe {timeframe} (TRAIN) '.center(60, '*'))
    print(f"* Profit / Loss  : {profit:.2f}")
    print(f"* Wins / Losses  : {wins} / {losses}")
    print(f"* Win Rate       : {(100 * (wins/(wins + losses)) if wins + losses > 0 else 0):.2f}%")

    # Get Reward from test data
    profit, wins, losses = get_result(test, solution[0], solution[1], solution[2])

    # Show the final result
    print(f' Result for timeframe {timeframe} (TEST) '.center(60, '*'))
    print(f"* Profit / Loss  : {profit:.2f}")
    print(f"* Wins / Losses  : {wins} / {losses}")
    print(f"* Win Rate       : {(100 * (wins/(wins + losses)) if wins + losses > 0 else 0):.2f}%")

    print("")
