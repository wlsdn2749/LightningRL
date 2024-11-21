import argparse
import os
import random
from collections import deque, namedtuple
from typing import Deque, List, Tuple

import gymnasium as gym
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lightning import Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import Dataset

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


class ReplayBuffer(Dataset):
    def __init__(self, replay_buffer, batch_size) -> None:
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size

    def __len__(self) -> int:
        return len(self.replay_buffer)

    def __getitem__(self, idx):
        """

        Args:
            idx: not used in this implementation


        Returns:
            tuple: A tuple containing:
                - torch.Tensor: Batch of states
                - torch.Tensor: Batch of actions
                - torch.Tensor: Batch of rewards
                - torch.Tensor: Batch of next states
                - torch.Tensor: Batch of done flags (boolean tensor)
        """
        transition = random.sample(self.replay_buffer, self.batch_size)
        batch = Transition(*zip(*transition))

        return (
            torch.tensor(batch.state),
            torch.tensor(batch.action),
            torch.tensor(batch.reward),
            torch.tensor(batch.new_state),
            torch.tensor(batch.done, dtype=torch.bool),
        )


class DQN(nn.Module):
    """
    A Deep Neural Networks Implementation

    :param state_size: gym observation_space's size
    :param action_siace: gym action_space's size
    """

    def __init__(self, state_size, action_size: int, n_quantiles: int = 32) -> None:
        super().__init__()
        self.action_size = action_size
        self.n_quantiles = n_quantiles
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_size * self.n_quantiles)  # output is |A| * N

    def forward(self, state: torch.Tensor) -> torch.tensor:
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x.view(state.shape[0], self.action_size, self.n_quantiles)  # [b, A, N]


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
        self.n_quantiles = self.hparams.n_quantiles
        self.net = DQN(self.state_size, self.action_size, self.n_quantiles)
        self.target_net = DQN(self.state_size, self.action_size, self.n_quantiles)
        self.replay_buffer: Deque[Transition] = deque(
            maxlen=self.hparams.buffer_capacity
        )
        self.state = self.env.reset()[0]
        self.gamma = self.hparams.gamma

        self.sync_rate = self.hparams.sync_rate
        self.batch_size = self.hparams.batch_size
        self.kappa = self.hparams.kappa
        self.register_buffer(
            "tau",
            torch.FloatTensor(
                [(2 * i + 1) / (2 * self.n_quantiles) for i in range(self.n_quantiles)]
            ),
        )

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
        self.populate(steps=self.hparams.warm_start_steps)  # TODO

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

            action_value = self.forward(state)
            action = torch.argmax(torch.mean(action_value, dim=2), dim=1).item()

        return action

    def step(self, device: str = "cpu") -> Tuple[float, bool]:
        # Epsilon Greedy
        action = self.get_action(device)

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
        reward, done = self.step(device)

        self.episode_reward += reward
        self.episode_steps += 1

        state_batch, action_batch, reward_batch, new_state_batch, done_batch = batch

        current_quantiles = torch.gather(
            self.net(state_batch),
            1,
            action_batch.unsqueeze(-1)
            .expand(self.batch_size, self.n_quantiles, -1)
            .transpose(1, 2),
        )  # [b, 1, N]

        with torch.no_grad():
            # 모든 Quantiles에 대해서 target_value 계산
            # T θj ← r + γθj(x′, a∗), ∀j
            new_quantiles = self.target_net(new_state_batch)  # [b, A, N]
            new_actions = torch.argmax(torch.mean(new_quantiles, dim=2), dim=1)  # [b]

            new_quantiles = torch.gather(
                new_quantiles,
                1,
                new_actions.unsqueeze(-1)
                .expand(self.batch_size, self.n_quantiles, -1)
                .transpose(1, 2),
            )
            new_quantiles[done_batch] = 0.0
            target_quantiles = (
                reward_batch.unsqueeze(1).unsqueeze(2) + self.gamma * new_quantiles
            )

        # Huber Loss 계산
        # 명시적인 논문 구현과정과 디버깅을 위해 torch.nn.Huberloss 사용하지 않음
        # 사용하려면 torch.nn.HuberLoss(target.. - curr.., delta=kappa)
        target_quantiles = target_quantiles.squeeze(1)
        current_quantiles = current_quantiles.squeeze(1)

        diff = target_quantiles.unsqueeze(-1) - current_quantiles.unsqueeze(-2)
        abs_diff = abs(diff)
        huber_loss = torch.where(
            abs_diff <= self.kappa,
            0.5 * pow(diff, 2),
            self.kappa * (abs_diff - 0.5 * self.kappa),
        )

        # Quantile Huber Loss 계산
        tau_diff = self.tau.unsqueeze(-2) - (diff.detach() < 0).float()
        loss = torch.mean(abs(tau_diff) * huber_loss)

        if self.global_step % self.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

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
        return ReplayBuffer(self.replay_buffer, batch_size=self.hparams.batch_size)

    def test_dataloader(self):
        return ReplayBuffer(self.replay_buffer, batch_size=self.hparams.batch_size)

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
        # how many quantile you use? classic=32, atari=200
        parser.add_argument("--n_quantiles", type=int, default=32)

        # Quantile Huber loss Hyperparameter
        parser.add_argument("--kappa", type=float, default=1.0)  #

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
        "--algo", type=str, default="QR-DQN", help="algorithm to use for training"
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
