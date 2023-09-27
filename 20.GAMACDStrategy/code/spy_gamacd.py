import numpy as np
import pandas as pd
import pandas_ta as ta
import pygad

from tqdm import tqdm

# Constants
DEBUG = 0
CASH = 10_000
SOLUTIONS = 30
GENERATIONS = 50
FILE_TRAIN = '../data/spy.train.csv.gz'
FILE_TEST = '../data/spy.test.csv.gz'
TREND_LEN = 7
MIN_TRADES_PER_DAY = 1
MAX_TRADES_PER_DAY = 10

# Configuration
np.set_printoptions(suppress=True)
pd.options.mode.chained_assignment = None

# Loading data, and split in train and test datasets
def get_data():

    train = pd.read_csv(FILE_TRAIN, compression='gzip')
    train['date'] = pd.to_datetime(train['date'])
    train.ta.ppo(close=train['close'], append=True)
    train = train.dropna().reset_index(drop=True)

    test = pd.read_csv(FILE_TEST, compression='gzip')
    test['date'] = pd.to_datetime(test['date'])
    test.ta.ppo(close=test['close'], append=True)
    test = test.dropna().reset_index(drop=True)

    train = train[train['date'] > (test['date'].max() - pd.Timedelta(365 * 10, 'D'))]

    return train, test

# Define fitness function to be used by the PyGAD instance
def fitness_func(self, solution, sol_idx):

    # Get Reward from train data
    reward, wins, losses, pnl = get_result(train, train_dates,
                                 solution[           :TREND_LEN*1],
                                 solution[TREND_LEN*1:TREND_LEN*2],
                                 solution[TREND_LEN*2:TREND_LEN*3],
                                 solution[TREND_LEN*3:TREND_LEN*4])

    if DEBUG:
        print(f'\n{reward:10.2f}, {pnl:10.2f}, {wins:6.0f}, {losses:6.0f}, {solution[TREND_LEN*1:TREND_LEN*2]}, {solution[TREND_LEN*3:TREND_LEN*4]}', end='')

    # Return the solution reward
    return reward

# Define a reward function
def get_result(df, business_dates, min_dist_buy, trend_buy, max_dist_sell, trend_sell, is_test=False):

    # Min/Max Trades
    min_trades = len(business_dates) * MIN_TRADES_PER_DAY
    max_trades = len(business_dates) * MAX_TRADES_PER_DAY

    # Buy & Sell Signals
    buy_mask = True
    sell_mask = True

    for i in range(0, len(min_dist_buy)):

        buy_mask = buy_mask & (df['PPOh_12_26_9'] > min_dist_buy[i])
        sell_mask = sell_mask & (df['PPOh_12_26_9'] < max_dist_sell[i])

        if i == 0: continue

        if trend_buy[i] > 0:
            buy_mask = buy_mask & (df['PPOh_12_26_9'].shift(i - 1) > df['PPOh_12_26_9'].shift(i))
        elif trend_buy[i] < 0:
            buy_mask = buy_mask & (df['PPOh_12_26_9'].shift(i - 1) < df['PPOh_12_26_9'].shift(i))

        if trend_sell[i] > 0:
            sell_mask = sell_mask & (df['PPOh_12_26_9'].shift(i - 1) > df['PPOh_12_26_9'].shift(i))
        elif trend_sell[i] < 0:
            sell_mask = sell_mask & (df['PPOh_12_26_9'].shift(i - 1) < df['PPOh_12_26_9'].shift(i))

    if buy_mask.sum() == 0: # Return if there are no buy signals
        return max(-999999, -len(df) + buy_mask.sum()), 0, 0, 0

    if sell_mask.sum() == 0: # Return if there are no sell signals
        return max(-999999, -len(df) + sell_mask.sum()), 0, 0, 0

    df['signal'] = np.where(buy_mask, 1, 0)
    df['signal'] = np.where(sell_mask, -1, df['signal'])

    # Remove all rows without operations, rows with the same consecutive operation, first row selling, and last row buying
    ops = df[df['signal'] != 0]
    ops = ops[ops['signal'] != ops['signal'].shift()]
    if (len(ops) > 0) and (ops.iat[0, -1] == -1): ops = ops.iloc[1:]
    if (len(ops) > 0) and (ops.iat[-1, -1] == 1): ops = ops.iloc[:-1]

    if len(ops) == 0: # Return if there are no operations
        return -min_trades, 0, 0, 0

    # Calculate P&L / operation
    ops['pnl'] = np.where(ops['signal'] == -1, (ops['close'] - ops['close'].shift()) * (CASH // ops['close'].shift()), 0)

    # Calculate total P&L, wins, and losses
    pnl = ops['pnl'].sum()
    wins = len(ops[ops['pnl'] > 0])
    losses = len(ops[ops['pnl'] < 0])

    # Calculate Expected Value
    valid_ops = ops[ops['pnl'] != 0]
    if len(valid_ops) == 0: # Return if there are no valid operations
        return -min_trades, 0, 0, 0

    if not is_test and (len(valid_ops) < min_trades):
        ev = -min_trades + len(valid_ops) # Penalize if there are less trades than the minimum allowed
    elif not is_test and (len(valid_ops) > max_trades):
        ev = -min_trades # Penalize if there are more trades than the maximum allowed
    else:
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
        ev = win_rate * ops[ops['pnl'] > 0]['pnl'].sum() - (1 - win_rate) * -ops[ops['pnl'] < 0]['pnl'].sum()

    return ev, wins, losses, pnl

# Get Train and Test data
train, test = get_data()

# Calculate Business days for train and test datasets
train_dates = train[['date']].set_index('date').resample('1D').max()
train_dates = train_dates[train_dates.index.dayofweek < 5]

test_dates = test[['date']].set_index('date').resample('1D').max()
test_dates = test_dates[test_dates.index.dayofweek < 5]

# Process data
print("".center(60, "*"))
print(f' PROCESSING DATA '.center(60, '*'))
print("".center(60, "*"))

with tqdm(total=GENERATIONS) as pbar:

    # Define Gene space based on configuration
    gene_space = []

    for i in range(TREND_LEN):
        gene_space.append({'low': -1, 'high': 1.1, 'step': 0.1})

    for i in range(TREND_LEN):
        gene_space.append({'low': -1, 'high': 2, 'step': 1})

    for i in range(TREND_LEN):
        gene_space.append({'low': -1, 'high': 1.1, 'step': 0.1})

    for i in range(TREND_LEN):
        gene_space.append({'low': -1, 'high': 2, 'step': 1})

    # Create Genetic Algorithm
    ga_instance = pygad.GA(num_generations=GENERATIONS,
                           num_parents_mating=5,
                           fitness_func=fitness_func,
                           sol_per_pop=SOLUTIONS,
                           num_genes=len(gene_space),
                           gene_space=gene_space,
                           parent_selection_type="sss",
                           crossover_type="single_point",
                           mutation_type="random",
                           mutation_by_replacement=True,
                           mutation_num_genes=2,
                           keep_parents=-1,
                           random_seed=42,
                           on_generation=lambda _: pbar.update(1),
                           )

    # Run the Genetic Algorithm
    ga_instance.run()

# Show details of the best solution.
solution, _, _ = ga_instance.best_solution()

print('\n')
print(f' Best Solution Parameters '.center(60, '*'))
print(f"Min Dist Buy    : {solution[           :TREND_LEN*1]}")
print(f"Trend Buy       : {solution[TREND_LEN*1:TREND_LEN*2]}")
print(f"Max Dist Sell   : {solution[TREND_LEN*2:TREND_LEN*3]}")
print(f"Trend Sell      : {solution[TREND_LEN*3:TREND_LEN*4]}")

# Get Reward from train data
reward, wins, losses, profit = get_result(train, train_dates,
                                          solution[           :TREND_LEN*1],
                                          solution[TREND_LEN*1:TREND_LEN*2],
                                          solution[TREND_LEN*2:TREND_LEN*3],
                                          solution[TREND_LEN*3:TREND_LEN*4],
                                          True)

print(f' Result (TRAIN) '.center(60, '*'))
print(f"* Reward         : {reward:.2f}")
print(f"* Profit / Loss  : {profit:.2f}")
print(f"* Wins / Losses  : {wins} / {losses}")
print(f"* Win Rate       : {(100 * (wins/(wins + losses)) if wins + losses > 0 else 0):.2f}%")

# Get Reward from test data
reward, wins, losses, profit = get_result(test, test_dates,
                                          solution[           :TREND_LEN*1],
                                          solution[TREND_LEN*1:TREND_LEN*2],
                                          solution[TREND_LEN*2:TREND_LEN*3],
                                          solution[TREND_LEN*3:TREND_LEN*4],
                                          True)

# Show the final result
print(f' Result (TEST) '.center(60, '*'))
print(f"* Reward         : {reward:.2f}")
print(f"* Profit / Loss  : {profit:.2f}")
print(f"* Wins / Losses  : {wins} / {losses}")
print(f"* Win Rate       : {(100 * (wins/(wins + losses)) if wins + losses > 0 else 0):.2f}%")