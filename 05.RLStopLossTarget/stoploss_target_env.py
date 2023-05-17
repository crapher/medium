import numpy as np
import gym
from gym import spaces

# Penalization 
NOOP_PENALIZATION = 0.01

# Exit Points
STOP_LOSS = 0.01
TARGET = 0.015

# Operations
HOLD = 0
BUY = 1

class StopLossTargetEnv(gym.Env):
        
    def __init__(self, observation_size, features, prices):

        # Data
        self.__features = features
        self.__prices = prices

        # Spaces 
        self.observation_space = spaces.Dict({
            'last_action': spaces.Discrete(2), 
            'observation': spaces.Box(low=np.NINF, high=np.PINF, shape=(observation_size, features.shape[1]), dtype=np.float32)})
        self.action_space = spaces.Discrete(2)

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
            self.__stop_loss = self.__open_price * (1 - STOP_LOSS)
            self.__target = self.__open_price * (1 + TARGET)
            
            self.__current_action = BUY
        elif self.__current_action == BUY:
            current_price = self.__prices[self.__current_tick]
            
            if current_price < self.__stop_loss or current_price > self.__target:
                step_reward = current_price - self.__open_price
                self.__current_profit += step_reward
                self.__current_action = HOLD
                
                if step_reward > 0:
                    self.__wins += 1
                else:
                    self.__losses += 1
        elif self.__current_action == HOLD:
            step_reward = NOOP_PENALIZATION

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

    def action_masks(self):
        
        # Allow to BUY only if the current position is HOLD
        mask = np.ones(self.action_space.n, dtype=bool)     
        mask[BUY] = self.__current_action == HOLD

        return mask
        
    def __get_observation(self):

        # If current tick over the last value in the feature array, the environment needs to be reset
        if self.__current_tick >= self.__end_tick:
            return None

        # Return the calculated observation
        return {
            'last_action': self.__current_action,
            'observation': self.__features[(self.__current_tick - self.__start_tick):self.__current_tick]
        }
