import gym
import numpy as np
from collections import deque
import time  # Added for reliable time tracking

class CustomCartPoleEnv(gym.Wrapper):
    def __init__(self, env, max_steps=400, loss_penalty=-10, success_reward=10):  # Changed max_points to max_steps
        super(CustomCartPoleEnv, self).__init__(env)
        self.max_steps = max_steps  # Number of steps to consider success
        self.loss_penalty = loss_penalty
        self.success_reward = success_reward
        self.steps = 0

    def reset(self, **kwargs):
        self.steps = 0
        observation, info = self.env.reset(**kwargs)
        self.frame_stack = deque([observation] * 4, maxlen=4)
        return np.array(self.frame_stack).flatten()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.steps += 1

        # Give reward of 1 for each successful step
        if not terminated and not truncated:
            reward = 1

        # End conditions
        if terminated or truncated:
            reward = self.loss_penalty
            terminated = True
            truncated = True
        elif self.steps >= self.max_steps:  # Success if agent reaches max_steps
            print(f"Success! Steps survived: {self.steps}")
            reward = self.success_reward
            terminated = True
            truncated = True

        self.frame_stack.append(observation)
        stacked_observation = np.array(self.frame_stack).flatten()
        return stacked_observation, reward, terminated, truncated, info

# Create environment
env = gym.make("CartPole-v1", render_mode=None)
env = CustomCartPoleEnv(env, max_steps=400)  # Reduce steps needed for success