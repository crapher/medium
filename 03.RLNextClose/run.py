### This file is the complete code of this Medium article
### https://medium.com/@diegodegese/reinforcement-learning-for-stock-trading-strategies-a-comprehensive-guide-c56677a9943
import math
import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from next_close_env import NextCloseEnv

# Read the data and generate the train and test dataframes
df = pd.read_csv('OIH_15T.csv.gz', compression='gzip')
train = df[df['date'] <= '2022-01-01']
test = df[df['date'] > '2022-01-01']

# Create 4 parallel train environments
env = make_vec_env(NextCloseEnv, seed=42, n_envs=4, env_kwargs={'observation_size': 26, 'closes': train['close'].values})

# Train the model
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000000)

# Save, remove, and reload the model (To be sure it works as expected)
model.save("ppo_next_close")
del model
model = PPO.load("ppo_next_close")

# Create a test environmant
env = NextCloseEnv(observation_size=26, closes=test['close'].values)

# Create the required variables for calculation
real = []
predicted = []
done = False

# Predict the test values with the trained model
obs = env.reset()
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)

    # Save the values to calculate the errors
    real.append(info['real_target'])
    predicted.append(info['agent_real_target'])

    print(f"AT: {info['agent_target']:6.3f} - CT: {info['calculated_target']:6.3f} - ART: {info['agent_real_target']:9.3f} - RT: {info['real_target']:9.3f}")

# Show results
real = np.array(real)
predicted = np.array(predicted)

mse = np.square(real - predicted).mean()
rmse = math.sqrt(mse)
mae = np.mean(np.abs(real - predicted))
mape = np.mean(np.abs((real - predicted) / real)) * 100

print(' RESULT '.center(56, '*'))
print(f"* Mean Square Error:              {mse:.3f}")
print(f"* Root Mean Square Error:         {rmse:.3f}")
print(f"* Mean Absolute Error:            {mae:.3f}")
print(f"* Mean Absolute Percentage Error: {mape:.3f} %")