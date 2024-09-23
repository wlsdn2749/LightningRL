import gymnasium as gym
import torch
import matplotlib.pyplot as plt
'''
    :Observation Space
        Discrete(8x8)
    
    :Action Space
        0: Move left
        1: Move down
        2: Move right
        3: Move up
'''
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)

q = torch.zeros((env.observation_space.n, env.action_space.n)) # Init 64x4 spaces

# Hyperparameters
discount_factor = 0.9
learning_factor = 0.2
episodes = 15000
epsilon = 1
epsilon_decay_rate = 0.0001
reward_per_episode = torch.zeros(episodes)

for idx in range(episodes):
    terminated = False
    truncated = False
    state, info = env.reset()
    
    while not terminated and not truncated:
        if torch.rand(1) < epsilon:
            action = env.action_space.sample()
        else:
            action = torch.argmax(q[state, :]).item()
            
        new_state, reward, terminated, truncated, info = env.step(action)
        
        # print(new_state, reward, terminated, truncated)
        q[state, action] = (1 - learning_factor) * q[state, action] + (
            learning_factor * (reward + discount_factor * torch.max(q[new_state, :]))
        )

        if reward == 1:
            reward_per_episode[idx] += 1
            
        state = new_state
    
    if idx % 1000 == 0:
        print(f"Episode {idx}, Average Reward: {torch.mean(reward_per_episode[max(0, idx-1000):idx]):.4f}")
        
    epsilon = max(epsilon - epsilon_decay_rate, 0)

env.close()

# Logging
sum_rewards = torch.zeros(episodes)
window_size = 1000
moving_average = torch.zeros(episodes - window_size + 1)
    
for t in range(len(moving_average)):
    moving_average[t] = torch.mean(reward_per_episode[t:t+window_size])

plt.figure(figsize=(12, 6))
plt.plot(range(window_size-1, episodes), moving_average)
plt.title("Moving Average of Rewards over Episodes")
plt.xlabel("Episode")
plt.ylabel("Average Reward (over {} episodes)".format(window_size))
plt.show()

