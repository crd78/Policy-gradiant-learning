import gym
import numpy as np
import torch
from Agent.Agent import Agent, FrameStack
from Env.env import env
import matplotlib.pyplot as plt

def main():
    # Initialize environment and agent
    n_actions = env.action_space.n
    input_shape = 64

    agent = Agent(input_shape=input_shape, n_actions=n_actions)
    frame_stack = FrameStack(k=4)

    num_episodes = 10000 
    reward_history = []
    success_history = []

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        frame_stack.reset(state)
        state = frame_stack.get_state()
        
        total_reward = 0
        done = False
        success = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Store reward before frame stack update
            agent.store_reward(reward)
            total_reward += reward
            
            # Update state
            frame_stack.append(next_state)
            state = frame_stack.get_state()
            
            if terminated or truncated:
                done = True
                if reward > env.loss_penalty:  # Fixed: using env.loss_penalty instead of self
                    success = True
                agent.train()

        reward_history.append(total_reward)
        success_history.append(success)

        if episode % 100 == 0:
            average_reward = np.mean(reward_history[-100:])
            success_rate = np.mean(success_history[-100:]) * 100
            print(f'Episode {episode}, Average Reward: {average_reward}, Success Rate: {success_rate:.2f}%')
            agent.save_model(average_reward)  # Save model if improved

    env.close()

    # Plot the rewards
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(reward_history)
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Training Progress - Total Reward')

    plt.subplot(1, 2, 2)
    plt.plot(success_history)
    plt.xlabel('Episodes')
    plt.ylabel('Success (1) / Failure (0)')
    plt.title('Training Progress - Success Tracking')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()