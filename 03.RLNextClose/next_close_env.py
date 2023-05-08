### This file is the complete code of this Medium article
### https://medium.com/@diegodegese/reinforcement-learning-for-stock-trading-strategies-a-comprehensive-guide-c56677a9943
import numpy as np
import gym
from gym import spaces

OBS_MIN_MAX = 0.05

class NextCloseEnv(gym.Env):

    def __init__(self, observation_size, closes):

        # Data
        self.__features = closes[:-1]
        self.__targets = closes[1:]

        # Spaces
        self.observation_space = spaces.Box(low=np.NINF, high=np.PINF, shape=(observation_size,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # Episode Management
        self.__start_tick = observation_size
        self.__end_tick = len(self.__targets)
        self.__current_tick = self.__end_tick

    def reset(self):

        # Reset the current tick pointer and return a new observation
        self.__current_tick = self.__start_tick
        return self.__get_observation()

    def step(self, action):

        # If current tick is over the last index in the feature array, the environment needs to be reset
        if self.__current_tick > self.__end_tick:
            raise Exception('The environment needs to be reset.')

        # Compute the step reward (Penalize the agent with the absolute difference between the real value and the prediction)
        step_reward = -abs(action[0] - self.__current_target)

        # Generate the custom info array with the real and predicted values
        info = {
            'agent_target': action[0],
            'calculated_target': self.__current_target,
            'agent_real_target': (action[0] * OBS_MIN_MAX + 1) * self.__current_avg,
            'real_target': self.__real_target}

        # Increase the current tick pointer, check if the environment is fully processed, and get a new observation
        self.__current_tick += 1
        done = self.__current_tick >= self.__end_tick
        obs = self.__get_observation()

        # Returns the observation, the step reward, the status of the environment, and the custom information
        return obs, step_reward, done, info

    def __get_observation(self):

        # If current tick over the last value in the feature array, the environment needs to be reset
        if self.__current_tick >= self.__end_tick:
            return None

        # Save the real observation and target values
        self.__real_observation = self.__features[(self.__current_tick - self.__start_tick):self.__current_tick]
        self.__real_target = self.__targets[self.__current_tick]

        # Generate a copy of the observation and target to avoid changing the original data
        self.__current_observation = self.__real_observation.copy()
        self.__current_target = self.__real_target.copy()

        # Calculate values between -1 and 1 for the new observation without leak any data
        self.__current_avg = np.mean(self.__current_observation)
        self.__current_observation = np.clip((self.__current_observation / self.__current_avg - 1) / OBS_MIN_MAX, -1, 1)
        self.__current_target = np.clip((self.__current_target / self.__current_avg - 1) / OBS_MIN_MAX, -1, 1)

        # Return the calculated observation
        return self.__current_observation
