import numpy as np
import pandas as pd
import pandas_ta as ta
import pygad
import pygad.kerasga

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sell_hold_buy_env import SellHoldBuyEnv

# Constants
OBS_SIZE = 32
FEATURES = 2
SOLUTIONS = 20
GENERATIONS = 50

# Loading data, and split in train and test datasets
df = pd.read_csv('OIH_1H.csv.gz', compression='gzip')
df.ta.bbands(close=df['close'], length=20, append=True)
df = df.dropna()
pd.options.mode.chained_assignment = None
df['high_limit'] = df['BBU_20_2.0'] + (df['BBU_20_2.0'] - df['BBL_20_2.0']) / 2
df['low_limit'] = df['BBL_20_2.0'] - (df['BBU_20_2.0'] - df['BBL_20_2.0']) / 2
df['close_percentage'] = np.clip((df['close'] - df['low_limit']) / (df['high_limit'] - df['low_limit']), 0, 1)
df['volatility'] = df['BBU_20_2.0'] / df['BBL_20_2.0'] - 1
train = df[df['date'] < '2022-01-01']
test = df[df['date'] >= '2022-01-01']

# Define fitness function to be used by the PyGAD instance
def fitness_func(self, solution, sol_idx):
    
    global model, observation_space_size, env
    
    # Set the weights to the model
    model_weights_matrix = pygad.kerasga.model_weights_as_matrix(model=model, weights_vector=solution)
    model.set_weights(weights=model_weights_matrix)

    # Run a prediction over the train data
    observation = env.reset()
    total_reward = 0

    done = False    
    while not done:
        state = np.reshape(observation, [1, observation_space_size])
        #q_values = model.predict(state, verbose=0)
        q_values = predict(state, model_weights_matrix)
        action = np.argmax(q_values[0])
        observation, reward, done, info = env.step(action)
        total_reward += reward
    
    # Print the reward and profit
    print(f"Solution {sol_idx:3d} - Total Reward: {total_reward:10.2f} - Profit: {info['current_profit']:10.3f}")

    if sol_idx == (SOLUTIONS-1):
        print("".center(60, "*"))
        
    # Return the solution reward
    return total_reward

def predict(X, W):
    X      = X.reshape((X.shape[0],-1))           #Flatten
    X      = X @ W[0] + W[1]                      #Dense
    X[X<0] = 0                                    #Relu
    X      = X @ W[2] + W[3]                      #Dense
    X[X<0] = 0                                    #Relu
    X      = X @ W[4] + W[5]                      #Dense
    X      = np.exp(X)/np.exp(X).sum(1)[...,None] #Softmax
    return X
    
# Create a train environmant
env = SellHoldBuyEnv(observation_size=OBS_SIZE, features=train[['close_percentage','volatility']].values, closes=train['close'].values)
observation_space_size = env.observation_space.shape[0] * FEATURES
action_space_size = env.action_space.n

# Create Model
model = Sequential()
model.add(Dense(16, input_shape=(observation_space_size,), activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(action_space_size, activation='linear'))
model.summary()

# Create Genetic Algorithm
keras_ga = pygad.kerasga.KerasGA(model=model, num_solutions=SOLUTIONS)

ga_instance = pygad.GA(num_generations=GENERATIONS,
                       num_parents_mating=5,
                       initial_population=keras_ga.population_weights,
                       fitness_func=fitness_func,
                       parent_selection_type="sss",
                       crossover_type="single_point",
                       mutation_type="random",
                       mutation_percent_genes=10,
                       keep_parents=-1)

# Run the Genetic Algorithm
ga_instance.run()

# Show details of the best solution.
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Fitness value of the best solution = {solution_fitness}")
print(f"Index of the best solution : {solution_idx}")

# Create a test environmant
env = SellHoldBuyEnv(observation_size=OBS_SIZE, features=test[['close_percentage','volatility']].values, closes=test['close'].values)

# Set the weights of the best solution to the model
best_weights_matrix = pygad.kerasga.model_weights_as_matrix(model=model, weights_vector=solution)
model.set_weights(weights=best_weights_matrix)

# Run a prediction over the test data
observation = env.reset()
total_reward = 0

done = False    
while not done:
    state = np.reshape(observation, [1, observation_space_size])
    #q_values = model.predict(state, verbose=0)
    q_values = predict(state, best_weights_matrix)
    action = np.argmax(q_values[0])
    observation, reward, done, info = env.step(action)
    total_reward += reward

# Show the final result
print(' RESULT '.center(60, '*'))
print(f"* Profit/Loss: {info['current_profit']:6.3f}")
print(f"* Wins: {info['wins']} - Losses: {info['losses']}")
print(f"* Win Rate: {100 * (info['wins']/(info['wins'] + info['losses'])):6.2f}%")
