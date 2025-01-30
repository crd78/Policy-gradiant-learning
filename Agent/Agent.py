import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class PolicyNetwork(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(PolicyNetwork, self).__init__()
        # Neural network architecture
        self.network = nn.Sequential(
            nn.Linear(input_shape, 128),  # input_shape is now 64
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.view(1, -1)  # Reshape to (batch_size, features)
        return self.network(x)

class Agent:
    def __init__(self, input_shape, n_actions, lr=1e-3):
        self.policy_net = PolicyNetwork(input_shape, n_actions)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = 0.99
        self.memory = []
        self.best_reward = float('-inf')

    def save_model(self, avg_reward):
        if avg_reward > self.best_reward:
            self.best_reward = avg_reward
            import os
            save_dir = 'saved_models'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(self.policy_net.state_dict(), f'{save_dir}/policy_net_best.pth')
            print(f"Model saved with average reward: {avg_reward:.2f}")

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        probs = self.policy_net(state)
        probs = probs.detach().numpy().flatten()  # Flatten to 1D array
        probs /= probs.sum()  # Ensure probabilities sum to 1
        action = np.random.choice(len(probs), p=probs)
        self.memory.append((state, action))
        return action

    def store_reward(self, reward):
        if len(self.memory) > 0:
            self.memory[-1] = (*self.memory[-1], reward)
        else:
            print("Warning: Attempted to store reward with empty memory.")

    def train(self):
        if len(self.memory) == 0:
            print("No memory to train on.")
            return

        rewards = [item[2] for item in self.memory]  # Get rewards from memory tuples
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        policy_losses = []
        for (state, action, _), G in zip(self.memory, returns):
            probs = self.policy_net(state)
            log_prob = torch.log(probs[0][action] + 1e-10)  # Add epsilon to prevent log(0)
            policy_losses.append(-log_prob * G)

        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_losses).sum()
        policy_loss.backward()
        self.optimizer.step()
        self.memory = []

class FrameStack:
    def __init__(self, k):
        self.k = k
        self.frames = deque(maxlen=k)

    def reset(self, frame):
        self.frames = deque([frame] * self.k, maxlen=self.k)

    def append(self, frame):
        self.frames.append(frame)

    def get_state(self):
        return np.array(self.frames).flatten()  # Return flattened array