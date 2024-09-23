import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
import random


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 64) 
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, state):
        x = self.fc1(state) # [bx64] -> [bx64]
        x = F.relu(x)
        x = self.fc2(x) # [bx64] -> [bx64]
        x = F.relu(x)
        x = self.fc3(x) # [bx64] -> [bx4]
        return x # [bx4]

def one_hot_encode(state, state_size):
    encoded = torch.zeros(state_size)
    encoded[state] = 1
    return encoded
    
def run():
        
    env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=False)
    
    transition = namedtuple('transition', ['state', 'action', 'reward', 'new_state', 'done'])
    #Initialize replay memory D to capacity N
    N = 10000
    D = deque(maxlen=N)
    
    #Initialize action-value function Q with random weights
    state_size = env.observation_space.n  # 64
    action_size = env.action_space.n # 4
    Q = QNetwork(state_size, action_size)
    
    episodes = 10000
    
    reward_per_episodes = torch.zeros(episodes)
    
    epsilon = 1
    epsilon_decay_rate = 0.0001
    batch_size = 32
    reward_factor = 0.9
    optimizer = optim.Adam(Q.parameters(), lr=0.001)

    for idx in range(episodes):
        
        # Initialise sequence s1 = {x1} 
        state, _ = env.reset()
        state = torch.FloatTensor(one_hot_encode(state, state_size))# .unsqueeze(0) # shape = [1x64]
        
        # ! DO NOT NEED preprocessed sequenced φ1 = φ(s1)
        terminated = False
        truncated = False
        
        while not terminated and not truncated:
            if torch.rand(1).item() < epsilon: # TODO Exploration
                action = env.action_space.sample() # TODO random action a
            else: # TODO Exploitation
                action = torch.argmax(Q(state)).item()
            
            # Execute action at in emulator and observe reward rt and image xt+1
            # Set st+1 = st, at, xt+1 
            # ! and DO NOT NEED preprocess φt+1 = φ(st+1)
            new_state, reward, terminated, truncated, info = env.step(action)
            
            new_state = torch.FloatTensor(one_hot_encode(new_state, state_size))# .unsqueeze(0) # shape = [1x64]
            done = terminated or truncated
            
            reward_per_episodes[idx] += reward
            
            # Store transition (φt, at, rt, φt+1) in D
            # Add 'done' to get new_state φt+1 is terminal or not
            D.append(transition(state, action, reward, new_state, done))
            
            # Sample random minibatch of transitions (φj, aj, rj, φj+1) from D
            if len(D) >= batch_size:
                transitions = random.sample(D, batch_size)
                batch = transition(*zip(*transitions)) # row 기준을 column 기준으로 변경 
                
                state_batch = torch.stack(batch.state) # [32x64]
                action_batch = torch.tensor(batch.action) #  [32]
                reward_batch = torch.tensor(batch.reward) # [32]
                new_state_batch = torch.stack(batch.new_state) # [32x64]
                done_batch = torch.tensor(batch.done, dtype=torch.bool) # [32]
                
                # minibatch's Q_values
                q_values = torch.gather(Q(state_batch), 1, action_batch.unsqueeze(-1))  
                
                new_state_values = torch.zeros(batch_size)
                 
                with torch.no_grad():
                    new_state_values[~done_batch] = torch.max(Q(new_state_batch[~done_batch]), dim=1)[0] # [32]
                    
                expected_q_values = reward_batch + (reward_factor * new_state_values * (~done_batch))
                
                # Perform a gradient descent step on (yj − Q(φj, aj; θ))2 according to equation 3
                loss = F.smooth_l1_loss(q_values, expected_q_values.unsqueeze(1))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            state = new_state
            epsilon = min(epsilon - epsilon_decay_rate, 0.1)
        
        if idx % 1000 == 0:
            print(f"Reward mean ~{idx} : {torch.mean(reward_per_episodes[idx-1000:idx])}")
            
if __name__ == "__main__":
    run()
    