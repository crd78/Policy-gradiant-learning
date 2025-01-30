from Env.env import CustomCartPoleEnv
import gym
from Agent.Agent import Agent, PolicyNetwork
import torch
import numpy as np
from collections import deque
import time  # Add time module

# Create environment with render mode
env = gym.make("CartPole-v1", render_mode="human")
env = CustomCartPoleEnv(env)

# Initialize agent with same parameters as training
input_shape = 64
n_actions = env.action_space.n
agent = Agent(input_shape=input_shape, n_actions=n_actions)

# Load the best model
model_path = 'saved_models/policy_net_best.pth'
agent.policy_net.load_state_dict(torch.load(model_path))
agent.policy_net.eval()

# Initialize frame stack
frame_stack = deque(maxlen=4)
state = env.reset()

# Initialize frame stack with initial state
for _ in range(4):
    frame_stack.append(state)

total_reward = 0
done = False

while not done:
    # Get current state
    current_state = np.array(frame_stack).flatten()
    
    # Select action using trained policy
    action = agent.select_action(current_state)
    
    # Execute action
    next_state, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    
    # Update frame stack
    frame_stack.append(next_state)
    
    done = terminated or truncated
    
    # Add delay between steps (0.02 seconds = 50 FPS)
    time.sleep(0.02)

print(f"Episode ended with total reward: {total_reward}")
env.close()