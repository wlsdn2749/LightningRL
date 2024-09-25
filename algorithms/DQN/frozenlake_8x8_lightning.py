from collections import deque, namedtuple
from typing import Tuple, Deque, List
import random
import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset

import lightning as L
from lightning import Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger

Transition = namedtuple(
    "Transition", ["state", "action", "reward", "new_state", "done"]
)


class MaxEpisodesCallback(Callback):
    def __init__(self, max_episodes: int = 1000):
        super().__init__()
        self.max_episodes = max_episodes

    def on_episode_end(self, trainer, pl_module) -> None:
        episode_count = trainer.callback_metrics.get("episodes", 0)
        if episode_count >= self.max_episodes:
            trainer.should_stop = True

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self.on_episode_end(trainer, pl_module)


class DQNLightning(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.env = gym.make(
            "FrozenLake-v1", desc=None, map_name="8x8", is_slippery=False
        )
        self.state_size = self.env.observation_space.n  # 64
        self.action_size = self.env.action_space.n  # 4
        self.state = self.env.reset()[0]
        self.net = DQN(self.state_size, self.action_size)
        self.target_net = DQN(self.state_size, self.action_size)
        self.replay_buffer: Deque[Transition] = deque(maxlen=10000)
        self.batch_size = 32
        self.epsilon = 1.0
        self.epsilon_decay_rate = 0.0001
        self.gamma = 0.9  # Discount Factor
        self.episode_reward = 0
        self.total_reward = 0
        self.episode_count = 0
        self.total_episode_steps = 0
        self.episode_steps = 0
        self.sync_rate = 10
        self.reward_list: List[float] = []
        self.populate()

    def forward(self, state):
        return self.net(state)

    def populate(self, steps: int = 10000):
        for _ in range(steps):
            self.step()

        # For logging
        for _ in range(100):
            self.reward_list.append(0)
        self.avg_reward = 0

    def step(self, device: str = "cpu") -> Tuple[float, bool]:
        # single step
        if torch.rand(1).item() < self.epsilon:
            action = self.env.action_space.sample()
        else:  # TODO Exploitation
            self.state = torch.tensor([self.state])
            if device not in ["cpu"]:
                self.state = self.state.cuda(device)
            action = torch.argmax(self.net(self.state)).item()

        new_state, reward, terminated, truncated, info = self.env.step(action)

        done = terminated or truncated

        transition = Transition(self.state, action, reward, new_state, done)
        self.replay_buffer.append(transition)

        self.state = new_state

        if done:
            self.state, _ = self.env.reset()

        return reward, done

    def training_step(self, batch, batch_idx):
        device = self.get_device(batch)
        self.epsilon = max(self.epsilon - self.epsilon_decay_rate, 0.01)
        reward, done = self.step(device)

        self.episode_reward += reward
        self.episode_steps += 1

        # Unpack the batch (which should be a transition tuple)
        state_batch, action_batch, reward_batch, new_state_batch, done_batch = batch

        # Current Q values
        q_values = torch.gather(
            self.net(state_batch), 1, action_batch.long().unsqueeze(-1)
        )

        # Target Q values
        with torch.no_grad():
            new_state_values = torch.zeros(
                self.batch_size, device=self.device, dtype=torch.float
            )
            new_state_values[~done_batch] = torch.max(
                self.target_net(new_state_batch[~done_batch]), dim=1
            )[0]
            expected_q_values = reward_batch + (
                self.gamma * new_state_values * (~done_batch)
            )

        loss = F.smooth_l1_loss(q_values, expected_q_values.unsqueeze(1))

        if done:
            self.total_reward = self.episode_reward
            self.reward_list.append(self.total_reward)
            self.avg_reward = sum(self.reward_list[-100:]) / 100
            self.episode_count += 1
            self.episode_reward = 0
            self.total_episode_steps += self.episode_steps
            self.episode_steps = 0

        if self.global_step % self.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        status = {
            "steps": self.global_step,
            "avg_reward": self.avg_reward,
            "total_reward": self.total_reward,
            "episodes": self.episode_count,
            "episode_steps": self.episode_steps,
            "epsilon": self.epsilon,
            "train_loss": loss,
        }

        self.log_dict(status, prog_bar=True)

        return loss

    def train_dataloader(self):
        return ReplayBuffer(self.replay_buffer, batch_size=self.batch_size)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.net.parameters(), lr=1e-3)
        return optimizer

    def get_device(self, batch):
        return batch[0].device.index if self.on_gpu else "cpu"


class ReplayBuffer(Dataset):
    def __init__(self, replay_buffer, batch_size) -> None:
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size

    def __len__(self) -> int:
        return len(self.replay_buffer)

    def __getitem__(
        self, idx
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        transitions = random.sample(self.replay_buffer, self.batch_size)
        batch = Transition(*zip(*transitions))

        return (
            torch.tensor(
                batch.state
            ),  # 원래 FloatTensor로 state를 받았을때는 stack 이였음
            torch.tensor(batch.action),
            torch.tensor(batch.reward),
            torch.tensor(
                batch.new_state
            ),  # 원래 FloatTensor로 state를 받았을때는 stack 이였음
            torch.tensor(batch.done, dtype=torch.bool),
        )


class DQN(nn.Module):
    def __init__(self, state_size, action_size) -> None:
        super().__init__()
        self.state_size = state_size
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state (torch.Tensor): 현재 상태를 나타내는 크기 [batch_size]

        Returns:
            torch.Tensor: 각 행동에 대한 Q값, 크기 [batch_size, action_size]
        """

        # state는 단일 vector로 들어오기때문에 [batch,1] -> [batch,state_size]로 one_hot encoding이 필요함
        x = F.one_hot(state, num_classes=self.state_size).float()
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x  # [bx4]


def one_hot_encode(state, state_size):
    encoded = torch.zeros(state_size)
    encoded[state] = 1
    return encoded


if __name__ == "__main__":
    wandb_logger = WandbLogger(
        project="lightning_dqn_frozenlake", save_dir="./wandb_logs/"
    )
    model = DQNLightning()

    trainer = Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else "auto",
        max_epochs=1000,
        callbacks=[MaxEpisodesCallback(max_episodes=1000)],
        logger=[wandb_logger],
    )

    trainer.fit(model)
