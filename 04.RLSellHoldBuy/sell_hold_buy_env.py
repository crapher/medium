### This file is the complete code of this Medium article
### https://medium.com/@diegodegese/reinforcement-learning-for-stock-trading-strategies-predicting-when-to-buy-and-sell-2ab5f542a41d
import numpy as np
import gym
from gym import spaces

# Normalization & Penalization 
OBS_MIN_MAX = 0.05
NOOP_PENALIZATION = 0.01

# Operations
SELL = 0
HOLD = 1
BUY = 2

class SellHoldBuyEnv(gym.Env):
        
    def __init__(self, observation_size, closes):

        # Data
        self.__features = closes
        self.__prices = closes

        # Spaces
        self.observation_space = spaces.Box(low=np.NINF, high=np.PINF, shape=(observation_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        # Episode Management
        self.__start_tick = observation_size
        self.__end_tick = len(self.__prices)
        self.__current_tick = self.__end_tick

        # Position Management
        self.__current_action = HOLD
        self.__current_profit = 0
        
    def reset(self):

        # Reset the current action and current profit
        self.__current_action = HOLD
        self.__current_profit = 0
        
        # Reset the current tick pointer and return a new observation
        self.__current_tick = self.__start_tick
        
        return self.__get_observation()

    def step(self, action):

        # If current tick is over the last index in the feature array, the environment needs to be reset
        if self.__current_tick > self.__end_tick:
            raise Exception('The environment needs to be reset.')

        # Compute the step reward (Penalize the agent if it is stuck doing anything)
        step_reward = 0
        if self.__current_action == HOLD and action == BUY:
            self.__open_price = self.__prices[self.__current_tick]
            self.__current_action = BUY
        elif self.__current_action == BUY and action == SELL:            
            step_reward = self.__prices[self.__current_tick] - self.__open_price
            self.__current_profit += step_reward
            self.__current_action = HOLD
        elif self.__current_action == HOLD:
            step_reward = NOOP_PENALIZATION

        # Generate the custom info array with the real and predicted values
        info = {
            'current_action': self.__current_action,
            'current_profit': self.__current_profit
        }

        # Increase the current tick pointer, check if the environment is fully processed, and get a new observation
        self.__current_tick += 1
        done = self.__current_tick >= self.__end_tick
        obs = self.__get_observation()

        # Returns the observation, the step reward, the status of the environment, and the custom information
        return obs, step_reward, done, info

    def action_masks(self):
        
        mask = np.ones(self.action_space.n, dtype=bool)
        
        # If current action is Buy, only allow to hold or sell
        if self.__current_action == BUY:
            mask[BUY] = False

        # If current action is Hold, only allow to hold or buy
        if self.__current_action == HOLD:
            mask[SELL] = False
        
        return mask
        
    def __get_observation(self):

        # If current tick over the last value in the feature array, the environment needs to be reset
        if self.__current_tick >= self.__end_tick:
            return None

        # Generate a copy of the observation to avoid changing the original data
        obs = self.__features[(self.__current_tick - self.__start_tick):self.__current_tick].copy()

        # Calculate values between -1 and 1 for the new observation without leak any data
        avg = np.mean(obs)
        obs = np.clip((obs / avg - 1) / OBS_MIN_MAX, -1, 1)

        # Return the calculated observation
        return obs
