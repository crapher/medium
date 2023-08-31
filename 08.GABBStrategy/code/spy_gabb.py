import numpy as np
import pandas as pd
import pandas_ta as ta
import pygad

from tqdm import tqdm

# Constants
CASH = 1000
SOLUTIONS = 30
GENERATIONS = 30
TRAIN_FILE = '../data/spy.train.csv.gz'
TEST_FILE = '../data/spy.test.csv.gz'

# Configuration
np.set_printoptions(suppress=True)
pd.options.mode.chained_assignment = None

# Loading data, and split in train and test datasets
def get_data():

    train = pd.read_csv(TRAIN_FILE, compression='gzip')
    train.ta.bbands(close=train['close'], length=10, append=True)
    train = train.dropna()
    train['high_limit'] = train['BBU_10_2.0'] + (train['BBU_10_2.0'] - train['BBL_10_2.0']) / 2
    train['low_limit'] = train['BBL_10_2.0'] - (train['BBU_10_2.0'] - train['BBL_10_2.0']) / 2
    train['close_percentage'] = np.clip((train['close'] - train['low_limit']) / (train['high_limit'] - train['low_limit']), 0, 1)
    train['bandwidth'] = np.clip(train['BBB_10_2.0'] / 100, 0, 1)

    test = pd.read_csv(TEST_FILE, compression='gzip')
    test.ta.bbands(close=test['close'], length=10, append=True)
    test = test.dropna()
    test['high_limit'] = test['BBU_10_2.0'] + (test['BBU_10_2.0'] - test['BBL_10_2.0']) / 2
    test['low_limit'] = test['BBL_10_2.0'] - (test['BBU_10_2.0'] - test['BBL_10_2.0']) / 2
    test['close_percentage'] = np.clip((test['close'] - test['low_limit']) / (test['high_limit'] - test['low_limit']), 0, 1)
    test['bandwidth'] = np.clip(test['BBB_10_2.0'] / 100, 0, 1)

    return train, test

# Define fitness function to be used by the PyGAD instance
def fitness_func(self, solution, sol_idx):

    # Get Reward from train data
    total_reward, _, _ = get_result(train, solution[0], solution[1], solution[2])

    # Return the solution reward
    return total_reward

# Define a reward function
def get_result(df, min_bandwidth, max_buy_perc, min_sell_perc):

    # Generate a copy to avoid changing the original data
    df = df.copy().reset_index(drop=True)

    # Buy Signal
    df['signal'] = np.where((df['bandwidth'] > min_bandwidth) & (df['close_percentage'] < max_buy_perc), 1, 0)

    # Sell Signal
    df['signal'] = np.where((df['close_percentage'] > min_sell_perc), -1, df['signal'])

    # Remove all rows without operations, rows with the same consecutive operation, first row selling, and last row buying
    result = df[df['signal'] != 0]
    result = result[result['signal'] != result['signal'].shift()]
    if (len(result) > 0) and (result.iat[0, -1] == -1): result = result.iloc[1:]
    if (len(result) > 0) and (result.iat[-1, -1] == 1): result = result.iloc[:-1]

    # Calculate the reward / operation
    result['total_reward'] = np.where(result['signal'] == -1, (result['close'] - result['close'].shift()) * (CASH // result['close'].shift()), 0)

    # Generate the result
    total_reward = result['total_reward'].sum()
    wins = len(result[result['total_reward'] > 0])
    losses = len(result[result['total_reward'] < 0])

    return total_reward, wins, losses

# Get Train and Test data
train, test = get_data()

# Process data
print("".center(60, "*"))
print(f' PROCESSING DATA '.center(60, '*'))
print("".center(60, "*"))

with tqdm(total=GENERATIONS) as pbar:

    # Create Genetic Algorithm
    ga_instance = pygad.GA(num_generations=GENERATIONS,
                           num_parents_mating=5,
                           fitness_func=fitness_func,
                           sol_per_pop=SOLUTIONS,
                           num_genes=3,
                           gene_space=[
                            {'low': 0, 'high': 1},
                            {'low': 0, 'high': 1, 'step': 0.01},
                            {'low': 0, 'high': 1, 'step': 0.01}],
                           parent_selection_type="sss",
                           crossover_type="single_point",
                           mutation_type="random",
                           mutation_num_genes=1,
                           keep_parents=-1,
                           random_seed=42,
                           on_generation=lambda _: pbar.update(1),
                           )

    # Run the Genetic Algorithm
    ga_instance.run()

# Show details of the best solution.
solution, solution_fitness, _ = ga_instance.best_solution()

print(f' Best Solution Parameters '.center(60, '*'))
print(f"Min Bandwidth    : {solution[0]:6.4f}")
print(f"Max Perc to Buy  : {solution[1]:6.2f}")
print(f"Min Perc to Sell : {solution[2]:6.2f}")

# Get Reward from train data
profit, wins, losses = get_result(train, solution[0], solution[1], solution[2])

print(f' Result (TRAIN) '.center(60, '*'))
print(f"* Profit / Loss  : {profit:.2f}")
print(f"* Wins / Losses  : {wins} / {losses}")
print(f"* Win Rate       : {(100 * (wins/(wins + losses)) if wins + losses > 0 else 0):.2f}%")

# Get Reward from test data
profit, wins, losses = get_result(test, solution[0], solution[1], solution[2])

# Show the final result
print(f' Result (TEST) '.center(60, '*'))
print(f"* Profit / Loss  : {profit:.2f}")
print(f"* Wins / Losses  : {wins} / {losses}")
print(f"* Win Rate       : {(100 * (wins/(wins + losses)) if wins + losses > 0 else 0):.2f}%")