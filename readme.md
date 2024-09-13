# Lightning Core RL
This repository is created for personal study and research purposes, containing code and examples of various reinforcement learning algorithms. Here, we will implement and experiment with different reinforcement learning algorithms using the [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) framework.

Most of these algorithms were based off the implementations found in [core_rl](https://github.com/djbyrne/core_rl)

Use **Wandb Logger** and Gym Monitor is default used. If you login wandb, you can see video and log on wandb pages

## Algorithms

### Model-free

#### Off Policy
- [x] Q-learning
- [x] SARSA 
- [x] DQN **(24-03-24)**
- [ ] Double DQN
- [ ] Dueling DQN
- [ ] Noisy DQN
- [ ] DQN with Prioritized Experience Replay
- [ ] N Step DQN
- [ ] DDPG
- [ ] TD3
- [ ] SAC

#### On Policy
- [ ] REINFORCE/Vanilla Policy Gradient
- [ ] A3C
- [ ] A2C
- [ ] PPO
- [ ] GAIL

### Model-based

- [x] Dyna / Dyna Q+


## Objectives
1. Implementation of various reinforcement learning algorithms
2. Explanation and example code for each algorithm
3. Visualization of experimental results and performance metrics

## Installation
```bash
poetry install
```

## Usage
Each algorithm is organized into separate directories, and you can refer to the README.md file in each algorithm's directory to learn about how to use and run them.


## Quick Start

Python 3.11 <br>
Lightning 2.1.1

```bash
poetry shell 
python dqn.py ...
```

## Contributions
While this repository was initiated for personal study and research, contributions and suggestions are always welcome. Whether it's bug reports, code improvements, or the implementation of new algorithms, any form of contribution is appreciated.

## License
The code in this repository is distributed under the [MIT License](https://github.com/wlsdn2749/Lightning-CoreRL/blob/master/LICENSE). Therefore, you are free to use it for both commercial and non-commercial purposes.

Feel free to utilize this repository to enhance your understanding of reinforcement learning, develop your own reinforcement learning algorithms, and conduct experiments in the field.