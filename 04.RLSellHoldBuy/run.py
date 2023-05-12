### This file is the complete code of this Medium article
### https://medium.com/@diegodegese/reinforcement-learning-for-stock-trading-strategies-predicting-when-to-buy-and-sell-2ab5f542a41d
import math
import numpy as np
import pandas as pd 

from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env

from sell_hold_buy_env import SellHoldBuyEnv

# Read the data and generate the train and test dataframes
df = pd.read_csv('OIH_15T.csv.gz', compression='gzip')
train = df[df['date'] <= '2022-01-01']
test = df[df['date'] > '2022-01-01']

# Create 4 parallel train environments
env = make_vec_env(SellHoldBuyEnv, seed=42, n_envs=4, env_kwargs={'observation_size': 26, 'closes': train['close'].values})

# Train the model
model = MaskablePPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000000)

# Save, remove, and reload the model (To be sure it works as expected)
model.save("maskableppo_sell_hold_buy")
del model
model = MaskablePPO.load("maskableppo_sell_hold_buy")

# Create a test environmant
env = SellHoldBuyEnv(observation_size=26, closes=test['close'].values)

# Create the required variables for calculation
done = False

# Predict the test values with the trained model
obs = env.reset()
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)

    print(f"Action: {info['current_action']} - Profit: {info['current_profit']:6.3f}")

# Show results
print(' RESULT '.center(56, '*'))
print(f"* Profit/Loss: {info['current_profit']:6.3f}")
