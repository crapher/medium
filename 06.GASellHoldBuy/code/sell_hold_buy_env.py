import numpy as np
import gym
from gym import spaces

# Operations
SELL = 0
HOLD = 1
BUY = 2

class SellHoldBuyEnv(gym.Env):
        
    def __init__(self, observation_size, features, closes):

        # Data
        self.__features = features
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
        self.__wins = 0
        self.__losses = 0
        
    def reset(self):

        # Reset the current action and current profit
        self.__current_action = HOLD
        self.__current_profit = 0
        self.__wins = 0
        self.__losses = 0
        
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
            
            if step_reward > 0:
                self.__wins += 1
            else:
                self.__losses += 1

        # Generate the custom info array with the real and predicted values
        info = {
            'current_action': self.__current_action,
            'current_profit': self.__current_profit,
            'wins': self.__wins,
            'losses': self.__losses
        }

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

        # Generate a copy of the observation to avoid changing the original data
        obs = self.__features[(self.__current_tick - self.__start_tick):self.__current_tick]

        # Return the calculated observation
        return obs
