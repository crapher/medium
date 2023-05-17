import math
import numpy as np
import pandas as pd 
import pandas_ta as ta

from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env

from stoploss_target_env import StopLossTargetEnv

# Read the data, generate the Percentage Price Oscillator (PPO) and create the train and test dataset
df = pd.read_csv('OIH_15T.csv.gz', compression='gzip')
df.ta.ppo(close=df['close'], append=True)
df.dropna(inplace=True)
train = df[(df['date'] >= '2018-01-01') & (df['date'] <= '2022-01-01')]
test = df[df['date'] > '2022-01-01']

# Create 4 parallel train environments
env = make_vec_env(StopLossTargetEnv, 
    seed=42, n_envs=4, 
    env_kwargs={'observation_size': 3, 'features': train[['PPO_12_26_9','PPOh_12_26_9','PPOs_12_26_9']].values, 'prices': train['close'].values})

# Train the model
model = MaskablePPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10_000_000)

# Save, remove, and reload the model (To be sure it works as expected)
model.save("maskableppo_stoploss_target")
del model
model = MaskablePPO.load("maskableppo_stoploss_target")

# Create a test environmant
env = StopLossTargetEnv(observation_size=3, features=test[['PPO_12_26_9','PPOh_12_26_9','PPOs_12_26_9']].values, prices=test['close'].values)

# Create the required variables for calculation
done = False

# Predict the test values with the trained model
obs = env.reset()
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, info = env.step(action)
    
    print(f"Action: {info['current_action']} - Profit: {info['current_profit']:6.3f}")

# Show results
print(' RESULT '.center(56, '*'))
print(f"* Profit/Loss: {info['current_profit']:6.3f}")
print(f"* Wins: {info['wins']} - Losses: {info['losses']}")
print(f"* Win Rate: {100 * (info['wins']/(info['wins'] + info['losses'])):6.2f}%")
