import gymnasium as gym
import torch

torch.linspace()


'''
    :Observation Space
    0: Cart Position (-4.8 ~ 4.8) 
    1: Cart Velocity (-inf ~ inf)
    2: Pole Angle (-0.418 ~ 0.418)
    3: Pole Angular Velocity (-inf ~ inf)
    
    :Action Space
    0: Push Cart to Left
    1: Push Cart to Right
'''

env = gym.make("CartPole-v1")


observation, info = env.reset(seed=42)

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()
        
env.close()
