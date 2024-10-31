import argparse
import os
import random
from collections import namedtuple
from typing import List, Tuple

import gymnasium as gym
import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lightning import Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset

import wandb

Transition = namedtuple(
    "Transition", ["state", "action", "reward", "new_state", "done"]
)

file_name = os.path.basename(os.path.abspath(__file__)).split(".")[0]


class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.

        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


class Monitor(gym.wrappers.RecordVideo):
    def __init__(
        self,
        env: gym.Env,
        video_folder: str,
        step_trigger=None,
        name_prefix: str = "rl-video",
        disable_logger=True,
    ):
        if step_trigger is None:
            step_trigger = lambda x: x % 2000 == 0

        super().__init__(
            env,
            video_folder,
            episode_trigger=None,
            step_trigger=step_trigger,
            name_prefix=name_prefix,
            disable_logger=disable_logger,
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


class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = torch.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)  # torch.empty()
        self.write = 0  # 다음 저장 위치
        self.size = 0  # 현재 저장된 경험 수

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total_priority(self) -> float:
        return self.tree[0].item()

    def add(self, priority: float, data: object):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.size < self.capacity:
            self.size += 1

    def update(self, idx: int, priority: float):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, object]:
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return (idx, self.tree[idx].item(), self.data[data_idx])

    def __len__(self) -> int:
        return self.size


class PrioritizedReplayMemory(Dataset):
    """
    A Prioritized Replay Memory by sum-tree

    :param replay_buffer: A list to store experience transitions, it is a tuple contains of state, action, reward, new_state_ done
    :param batch_size: The number of samples to return in each batch

    """

    def __init__(self, capacity: int, alpha=0.6, epsilon=1e-5):
        """
        :param capacity: Maximum size of Replay Memory
        :param alpha: Amount of Scaling Priority (0: DQN ~ 1: more PER)
        """
        super().__init__()
        self.alpha = alpha
        self.epsilon = epsilon
        self.tree = SumTree(capacity)
        self.beta = 0.4
        self.beta_increment_per_sampling = 0.001

    def sample(self, batch_size):
        """
        :param beta: Amount of Scaling Importance Weight
        :return: (Sampled Experiences, Indices, Importance Weights)
        """
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total_priority() / batch_size

        self.beta = np.min([1.0, self.beta + self.beta_increment_per_sampling])

        for i in range(batch_size):
            # s = min(torch.rand(1).item() * segment + segment * i, self.tree.total_priority() - self.epsilon) # 여기서 .get(s) 값이 계속 0.999를 초과하는 경우..
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, priority, data = self.tree.get(s)

            # 최대 범위를 초과하지 않도록 마지막 데이터로 제한 <- 안정적인 학습을 제한할 수 있음
            if data == 0:
                idx = self.tree.size - 1
                priority, data = self.tree.tree[idx].item(), self.tree.data[idx]

            batch.append(data)
            indices.append(idx)
            priorities.append(priority)

        priorities = torch.tensor(priorities, dtype=torch.float32)
        total_priority = self.tree.total_priority()

        probabilities = priorities / total_priority
        weights = (self.tree.size * probabilities) ** (-self.beta)
        weights /= weights.max()  # Importance sampling

        batch = Transition(*zip(*batch))

        return (
            torch.tensor(batch.state),
            torch.tensor(batch.action),
            torch.tensor(batch.reward),
            torch.tensor(batch.new_state),
            torch.tensor(batch.done, dtype=torch.bool),
            torch.tensor(indices),
            weights,
        )

    def add(self, transition, td_error):
        """
        Add priority to new transition
        :param transition: (state, action, reward, next_state, done)
        :param td_error: TD error
        """
        priority = (abs(td_error) + self.epsilon) ** self.alpha
        self.tree.add(priority, transition)

    def update_priorities(self, indices, td_errors):
        """
        After training, update the priority based on new td_error
        """
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error.item()) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)


class DummyDataset(Dataset):
    """
    A dummy dataset to satisfy the PyTorch Lightning `train_dataloader` requirement.
    This dataset will not be used for actual sampling.
    """

    def __len__(self):
        return 10000  # Random large number

    def __getitem__(self, index):
        return 0  # Return a placeholder value


class DuelingDQN(nn.Module):
    """
    A Deep Neural Networks Implementation

    :param state_size: gym observation_space's size
    :param action_siace: gym action_space's size
    """

    def __init__(
        self,
        state_size,  # ! NEED
        action_size: int,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)

        # Value stream this is Beta
        # self.value_fc = nn.Linear(256, 128)
        self.value = nn.Linear(256, 1)  # Output is scalar V(s)

        # Advantage stream this is Alpha
        # self.advantage_fc = nn.Linear(256, 128)
        self.advantage = nn.Linear(256, action_size)  # Output is A(s, a)

    def forward(self, state: torch.Tensor) -> torch.tensor:
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)

        # value and advantage
        v = self.value(x)
        a = self.advantage(x)
        mean_dim = 1 if a.dim() > 1 else 0
        q = v + a - a.mean(dim=mean_dim, keepdim=True)
        return q


class DQNLightning(L.LightningModule):
    def __init__(self, hparams) -> None:
        """
        DQN Training Class

        Hyperparams by using https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml
        """
        super().__init__()
        self.save_hyperparameters(hparams)
        self.env = gym.make(id=self.hparams.env, render_mode=self.hparams.render_mode)
        self.env = Monitor(
            self.env, video_folder=f"video/{file_name}", disable_logger=True
        )
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.net = DuelingDQN(self.state_size, self.action_size)
        self.target_net = DuelingDQN(self.state_size, self.action_size)
        self.replay_buffer = PrioritizedReplayMemory(self.hparams.buffer_capacity)
        self.state = self.env.reset()[0]
        self.gamma = self.hparams.gamma
        self.sync_rate = self.hparams.sync_rate
        self.batch_size = self.hparams.batch_size

        # Create the schedule for exploration starting from 1.
        exploration_fraction = self.hparams.exploration_fraction
        exploration_final_eps = self.hparams.exploration_final_eps
        total_timesteps = self.hparams.max_steps
        self.exploration = LinearSchedule(
            schedule_timesteps=int(exploration_fraction * total_timesteps),
            initial_p=1.0,
            final_p=exploration_final_eps,
        )
        # For Debug
        self.total_reward = 0
        self.episode_reward = 0
        self.total_episode_steps = 0
        self.episode_steps = 0
        self.episode_count = 0
        self.reward_list: List[float] = []
        self.avg_reward = 0

        # Warm Up
        self.populate(steps=self.hparams.warm_start_steps)

    def forward(self, state) -> torch.tensor:
        return self.net(state)

    def populate(self, steps: int = 10000) -> None:
        for _ in range(steps):
            self.step()

        for _ in range(100):
            self.reward_list.append(0)
        self.avg_reward = 0

    def get_action(self, device) -> int:
        state = self.state
        update_eps = self.exploration.value(self.global_step)

        if torch.rand(1).item() < update_eps:
            action = self.env.action_space.sample()

        else:
            state = torch.tensor([state])
            if device not in ["cpu"]:
                state = state.cuda(device)

            action = torch.argmax(self.forward(state)).item()

        return action

    def step(self, device: str = "cpu") -> Tuple[float, bool]:
        action = self.get_action(device)

        # 환경과 상호작용 (새로운 상태, 보상 등 반환)
        new_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        state_tensor = torch.tensor(self.state, dtype=torch.float32, device=device)

        # 현재 상태에서의 Q-value (정책 네트워크 사용)
        current_q_value = self.net(state_tensor)[action]

        new_state_tensor = torch.tensor(new_state, dtype=torch.float32, device=device)

        # 다음 상태에서의 최대 Q-value (타겟 네트워크 사용)
        with torch.no_grad():
            next_q_value = torch.max(self.target_net(new_state_tensor))

        # 종료된 경우, 다음 상태의 Q-value는 0으로 설정
        if done:
            next_q_value = torch.tensor(0.0, device=device)

        # TD 오차 계산
        td_error = reward + self.gamma * next_q_value - current_q_value

        # 경험 저장 (transition에 TD 오차 추가)
        transition = Transition(self.state, action, reward, new_state, done)

        # TD 오차를 기반으로 Replay Buffer에 추가 (TD 오차 사용)
        self.replay_buffer.add(transition, td_error.item())

        # 상태 업데이트
        self.state = new_state

        # 에피소드 종료 시 환경을 리셋
        if done:
            self.state, _ = self.env.reset()

        return reward, done

    def training_step(self, batch, batch_idx):
        device = self.get_device(batch)
        reward, done = self.step(device)

        self.episode_reward += reward
        self.episode_steps += 1

        *transitions, indices, weights = self.replay_buffer.sample(self.batch_size)

        state_batch, action_batch, reward_batch, new_state_batch, done_batch = (
            transitions
        )

        state_batch = state_batch.to(device)
        action_batch = action_batch.to(device)
        reward_batch = reward_batch.to(device)
        new_state_batch = new_state_batch.to(device)
        done_batch = done_batch.to(device)
        weights = weights.to(device)

        # policy network에서 Q(s, a; θ) 계산
        # 이때 action_batch의 unsqueeze(-1)는 [1xb] -> [bx1]로 바주어
        # self.net(state_batch)의 특정 인덱스만 추출하기 위함

        current_q_values = torch.gather(
            self.net(state_batch), 1, action_batch.long().unsqueeze(-1)
        )

        with torch.no_grad():
            # argmax_a Q(s', a; θ) 를 통해 actions를 구함
            # 여기서 θ는 policy network의 파라미터,
            next_actions = self.net(new_state_batch).argmax(1).unsqueeze(-1)

            # Q(s', argmax_a Q(s', a; θ); θ-) 계산
            # 여기서 θ- 는 타겟 네트워크의 파라미터를 나타냄
            next_q_values = torch.gather(
                self.target_net(new_state_batch), 1, next_actions
            ).squeeze(-1)
            next_q_values[done_batch] = 0.0
            next_q_values = next_q_values.detach()

        # target Q 값 계산: y = r + γ * Q(s', argmax_a Q(s', a; θ); θ^-)
        target_q_values = reward_batch + self.gamma * next_q_values

        # Beta가 기본값으로 1이므로 Huberloss와 완전히 동일한 결과
        loss = (
            F.smooth_l1_loss(
                current_q_values, target_q_values.unsqueeze(1), reduction="none"
            )
            * weights
        ).mean()

        if self.global_step % self.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        td_errors = torch.abs(target_q_values.unsqueeze(1) - current_q_values).detach()
        self.replay_buffer.update_priorities(indices, td_errors)

        if done:
            self.total_reward = self.episode_reward
            self.reward_list.append(self.total_reward)
            self.avg_reward = sum(self.reward_list[-100:]) / 100
            self.episode_count += 1
            self.episode_reward = 0
            self.total_episode_steps += self.episode_steps
            self.episode_steps = 0

        status = {
            "steps": self.global_step,
            "avg_reward": self.avg_reward,
            "total_reward": self.total_reward,
            "episodes": self.episode_count,
            "episode_steps": self.episode_steps,
            "epsilon": self.exploration.value(self.global_step),
            "train_loss": loss,
        }

        self.log_dict(status, prog_bar=True)

        return loss

    def train_dataloader(self):
        """
        Return a dummy dataloader to satisfy Lightning requirements.
        """
        dataset = DummyDataset()
        return DataLoader(dataset, batch_size=1)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.net.parameters(), lr=self.hparams.lr)
        return optimizer

    def get_device(self, batch):
        return batch[0].device.index if self.on_gpu else "cpu"

    @staticmethod
    def add_model_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        parser = parent_parser.add_argument_group("DQNLightning")
        parser.add_argument("--buffer_capacity", type=int, default=100000)
        parser.add_argument("--exploration_fraction", type=float, default=0.16)
        parser.add_argument("--exploration_final_eps", type=float, default=0.02)
        parser.add_argument("--sync_rate", type=int, default=1000)
        parser.add_argument("--warm_start_steps", type=int, default=10000)
        return parent_parser


def get_project_name_counter(base_project_name) -> int:
    api = wandb.Api()
    user_name = api.default_entity
    runs = api.runs(f"{user_name}/{base_project_name}")
    try:
        run_count = len(runs)
    except Exception:
        return 0

    return run_count


def add_base_args(parent_parser) -> argparse.ArgumentParser:
    """
    Adds common arguments for reinforcement learning algorithms.

    This function configures arguments are findtuned for a CartPole environment,
    with common settings for learning rate, batch size, and other parameters.

    Args:
        parent_parser: (ArgumentParser): The parent parser to add arguments to.

    Returns:
        ArgumentParser: A parser with added arguments for learning

    Arguments:
        batch_size (int): Size of the training batches. Default is 32.
        lr (float): Learning rate for optimizer. Default is 1e-4.
        alpha (float): Alpha for prioritized experience replay. Default is 0.1.
        env (str): Gym environment ID. Default is 'CartPole-v1'.
        render_mode (str): Mode for rendering ('human', 'rgb_array', etc.). Default is 'rgb_array'.
        gamma (float): Discount factor for future rewards. Default is 0.99.
        episode_length (int): Max steps in each episode. Default is 500.
        max_steps (int): Max training steps. Default is 500000.
        devices (int): Number of devices to use. Default is 1.
        seed (int): Random seed for reproducibility. Default is 3721.
        strategy (str): Distributed training strategy. Default is 'auto'.
    """
    arg_parser = argparse.ArgumentParser(parent_parser)

    arg_parser.add_argument(
        "--algo", type=str, default="DuelingDQN", help="algorithm to use for training"
    )
    arg_parser.add_argument(
        "--batch_size", type=int, default=32, help="size of the batches"
    )
    arg_parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    arg_parser.add_argument("--alpha", type=float, default=0.1, help="alpha")
    arg_parser.add_argument(
        "--env", type=str, default="CartPole-v1", help="gym environment tag"
    )
    arg_parser.add_argument(
        "--render_mode",
        type=str,
        default="rgb_array",
        help="gym render_mode | human, rgb_array, ansl, None",
    )
    arg_parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    arg_parser.add_argument(
        "--episode_length", type=int, default=500, help="max length of an episode"
    )
    # arg_parser.add_argument("--max_episode_reward", type=int, default=18,
    #                         help="max episode reward in the environment")
    arg_parser.add_argument(
        "--max_steps", type=int, default=500000, help="max steps to train the agent"
    )
    # arg_parser.add_argument("--n_steps", type=int, default=4,
    #                         help="how many steps to unroll for each update")
    arg_parser.add_argument(
        "--devices", type=int, default=1, help="number of devices to use for training"
    )
    arg_parser.add_argument(
        "--seed", type=int, default=3721, help="seed for training run"
    )
    arg_parser.add_argument(
        "--strategy",
        type=str,
        default="auto",
        help="distributed backend strategy to be used by lightning",
    )
    return arg_parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_base_args(parser)
    parser = DQNLightning.add_model_specific_args(parser)
    args = parser.parse_args()
    model = DQNLightning(args)

    base_project_name = f"LightningRL-algorithms_{args.algo}"
    version = get_project_name_counter(base_project_name)
    wandb.init(
        monitor_gym=True,
        save_code=True,
        project=base_project_name,
        name=f"{args.env}-{args.algo}-{version}",
    )
    wandb_logger = WandbLogger(save_dir="./wandb/")

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1, monitor="avg_reward", mode="max"
    )

    trainer = Trainer(
        accelerator=args.strategy,
        devices=args.devices if torch.cuda.is_available() else "auto",
        max_steps=args.max_steps,
        val_check_interval=0.25,
        callbacks=checkpoint_callback,
        logger=[wandb_logger],
    )

    trainer.fit(model)

    wandb.finish()
