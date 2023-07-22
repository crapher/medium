import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Constant to have a value between -1 and 1
SCALE=1500

class NextTrendEnv(gym.Env):

    def __init__(self, observation_size, closes):

        # Data
        self.__features = closes[:-1]
        self.__targets = closes[1:]

        # Spaces
        self.observation_space = spaces.Box(low=-1, high=1, shape=(observation_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

        # Episode Management
        self.__start_tick = observation_size
        self.__end_tick = len(self.__targets)
        self.__current_tick = self.__end_tick

    def reset(self, seed=None, options=None):

        # Reset the current tick pointer and return a new observation
        self.__current_tick = self.__start_tick

        return self.__get_observation(), None

    def step(self, action):

        # If current tick is over the last index in the feature array, the environment needs to be reset
        if self.__current_tick > self.__end_tick:
            raise Exception('The environment needs to be reset.')

        # Assuming that the model returns 0 for downtrend and 1 for uptrend, it replaces a 0 with -1 for easier comparison.
        action = -1 if action == 0 else action

        # Compute the step reward (-1 if the model value is different to the target or 0 if the value is the same)
        step_reward = -1 if action != self.__target else 0

        # Generate the custom info array with the real and predicted values
        info = {
            'agent_target': action,
            'real_target': self.__target}

        # Increase the current tick pointer, check if the environment is fully processed, and get a new observation
        self.__current_tick += 1
        terminated = self.__current_tick >= self.__end_tick
        truncated = False
        obs = self.__get_observation()

        # Returns the observation, the step reward, the status of the environment, and the custom information
        return obs, step_reward, terminated, truncated, info

    def __get_observation(self):

        # If current tick over the last value in the feature array, the environment needs to be reset
        if self.__current_tick >= self.__end_tick:
            return None

        # Generate the observation (and scale it) and the target value
        self.__observation = self.__features[(self.__current_tick - self.__start_tick):self.__current_tick]
        self.__observation = np.clip(self.__observation / SCALE, -1, 1)
        self.__target = np.where(self.__targets[self.__current_tick] > self.__targets[self.__current_tick - 1], 1, -1)
        
        # Return the calculated observation
        return self.__observation
