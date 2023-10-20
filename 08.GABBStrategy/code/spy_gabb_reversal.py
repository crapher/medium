import numpy as np
import pandas as pd
import pandas_ta as ta
import pygad

from tqdm import tqdm

# Constants
CASH = 1000
SOLUTIONS = 30
GENERATIONS = 50
TRAIN_FILE = '../data/spy.train.csv.gz'
TEST_FILE = '../data/spy.test.csv.gz'

# Configuration
np.set_printoptions(suppress=True)
pd.options.mode.chained_assignment = None

# Loading data, and split in train and test datasets
def get_data():

    train = pd.read_csv(TRAIN_FILE, compression='gzip')
    train['date'] = pd.to_datetime(train['date'])
    train = train.dropna()

    test = pd.read_csv(TEST_FILE, compression='gzip')
    test['date'] = pd.to_datetime(test['date'])
    test = test.dropna()

    return train, test

# Define fitness function to be used by the PyGAD instance
def fitness_func(self, solution, sol_idx):

    # Get Reward from train data
    reward, _, _, _ = get_result(train, solution[0], solution[1], solution[2], solution[3])

    # Return the solution reward
    return reward

# Define a reward function
def get_result(df, buy_length, buy_std, sell_length, sell_std, is_test=False):

    # Round to 2 digit to avoid the Bollinger bands function to generate weird field names
    buy_std = round(buy_std, 2)
    sell_std = round(sell_std, 2)

    # Generate suffixes for Bollinger bands fields
    buy_suffix = f'{int(buy_length)}_{buy_std}'
    sell_suffix = f'{int(sell_length)}_{sell_std}'

    # Generate a copy to avoid changing the original data
    df = df.copy().reset_index(drop=True)

    # Calculate Bollinger bands based on parameters
    if not f'BBU_{buy_suffix}' in df.columns:
        df.ta.bbands(close=df['close'], length=buy_length, std=buy_std, append=True)
    if not f'BBU_{sell_suffix}' in df.columns:
        df.ta.bbands(close=df['close'], length=sell_length, std=sell_std, append=True)
    df = df.dropna()

    # Buy Signal
    df['signal'] = np.where(df['close'] < df[f'BBL_{buy_suffix}'], 1, 0)

    # Sell Signal
    df['signal'] = np.where(df['close'] > df[f'BBU_{sell_suffix}'], -1, df['signal'])

    # Remove all rows without operations, rows with the same consecutive operation, first row selling, and last row buying
    result = df[df['signal'] != 0]
    result = result[result['signal'] != result['signal'].shift()]
    if (len(result) > 0) and (result.iat[0, -1] == -1): result = result.iloc[1:]
    if (len(result) > 0) and (result.iat[-1, -1] == 1): result = result.iloc[:-1]

    # Calculate the reward & result / operation
    result['reward'] = np.where(result['signal'] == -1, (result['close'] - result['close'].shift()) * (CASH // result['close'].shift()), 0)
    result['wins'] = np.where(result['reward'] > 0, 1, 0)
    result['losses'] = np.where(result['reward'] < 0, 1, 0)

    # Generate window and filter windows without operations
    result_window = result.set_index('date').resample('3M').agg(
        {'close':'last','reward':'sum','wins':'sum','losses':'sum'}).reset_index()

    min_operations = 252 # 1 Year
    result_window = result_window[(result_window['wins'] + result_window['losses']) != 0]

    # Generate the result
    wins = result_window['wins'].mean() if len(result_window) > 0 else 0
    losses = result_window['losses'].mean() if len(result_window) > 0 else 0
    reward = result_window['reward'].mean() if (min_operations < (wins + losses)) or is_test else -min_operations + (wins + losses)
    pnl = result_window['reward'].sum()

    return reward, wins, losses, pnl

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
                           num_genes=4,
                           gene_space=[
                            {'low': 1, 'high': 200, 'step': 1},
                            {'low': 0.1, 'high': 3, 'step': 0.01},
                            {'low': 1, 'high': 200, 'step': 1},
                            {'low': 0.1, 'high': 3, 'step': 0.01}],
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
print(f'Buy Length    : {solution[0]:.0f}')
print(f'Buy Std       : {solution[1]:.2f}')
print(f'Sell Length   : {solution[2]:.0f}')
print(f'Sell Std      : {solution[3]:.2f}')

# Get result from train data
reward, wins, losses, pnl = get_result(train, solution[0], solution[1], solution[2], solution[3])

# Show the train result
print(f' Result (TRAIN) '.center(60, '*'))
print(f'* Reward                   : {reward:.2f}')
print(f'* Profit / Loss (B&H)      : {(train["close"].iloc[-1] - train["close"].iloc[0]) * (CASH // train["close"].iloc[0]):.2f}')
print(f'* Profit / Loss (Strategy) : {pnl:.2f}')
print(f'* Wins / Losses            : {wins:.2f} / {losses:.2f}')
print(f'* Win Rate                 : {(100 * (wins/(wins + losses)) if wins + losses > 0 else 0):.2f}%')

# Get result from test data
reward, wins, losses, pnl = get_result(test, solution[0], solution[1], solution[2], solution[3], True)

# Show the test result
print(f' Result (TEST) '.center(60, '*'))
print(f'* Reward                   : {reward:.2f}')
print(f'* Profit / Loss (B&H)      : {(test["close"].iloc[-1] - test["close"].iloc[0]) * (CASH // test["close"].iloc[0]):.2f}')
print(f'* Profit / Loss (Strategy) : {pnl:.2f}')
print(f'* Wins / Losses            : {wins:.2f} / {losses:.2f}')
print(f'* Win Rate                 : {(100 * (wins/(wins + losses)) if wins + losses > 0 else 0):.2f}%')
