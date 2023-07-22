import math
import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from sklearn.metrics import confusion_matrix, classification_report
from next_trend_env import NextTrendEnv

OBSERVATION_SIZE=2

# Read the data and generate the train and test dataframes
df = pd.read_csv('../data/NYSI.csv.gz', compression='gzip')
train = df[df['date'] <= '2020-06-01']
validation = df[(df['date'] > '2020-06-01') & (df['date'] <= '2022-01-01')]
test = df[df['date'] > '2022-01-01']

# Create 4 parallel train environments
env = make_vec_env(NextTrendEnv, seed=42, n_envs=4, env_kwargs={'observation_size': OBSERVATION_SIZE, 'closes': train['value'].values})

# Create a validation environment
eval_env = NextTrendEnv(observation_size=OBSERVATION_SIZE, closes=validation['value'].values)
eval_callback = EvalCallback(eval_env, best_model_save_path="./", eval_freq=2*len(train), deterministic=True, render=False)

# Train the model
model = PPO("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=1_000_000, callback=eval_callback, progress_bar=True)

# Remove, and reload the best model (To be sure it works as expected)
del model
model = PPO.load("best_model")

# Create a test environment
env = NextTrendEnv(observation_size=OBSERVATION_SIZE, closes=test['value'].values)

# Create the required variables for calculation
real = []
predicted = []
terminated = False

# Predict the test values with the trained model
obs, _ = env.reset()
while not terminated:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, terminated, truncated, info = env.step(action)

    # Save the values to calculate the errors
    real.append(info['real_target'])
    predicted.append(info['agent_target'])

# Show results
real = np.array(real)
predicted = np.array(predicted)

print(' RESULT TEST '.center(56, '*'))
print('* Confusion Matrix (Top: Predicted - Left: Real)')
print(confusion_matrix(real, predicted))
print('* Classification Report')
print(classification_report(real, predicted))
