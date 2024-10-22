import os
import yaml
import random

import torch
import gym

import numpy as np
import torch.nn as nn

from dataclasses import fields, replace
from typing import Optional, Tuple, Union, List
from net import Scalar


TensorBatch = List[torch.Tensor]


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wrap_env(
        env: gym.Env,
        state_mean: Union[np.ndarray, float] = 0.0,
        state_std: Union[np.ndarray, float] = 1.0,
        reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        return (
                       state - state_mean
               ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


def set_seed(
        seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


@torch.no_grad()
def eval_actor(
        env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int) -> np.ndarray:
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        while not done:
            action = actor.act(state, device)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards)


@torch.no_grad()
def eval_actor_explore(
        env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int) -> np.ndarray:
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        while not done:
            state_tensor = torch.FloatTensor(np.array([state])).to(device)
            action = actor(state_tensor)[0].detach().cpu().numpy().flatten()
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards)


@torch.no_grad()
def eval_deterministic_actor_explore(env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int,
                                     explore_noise, action_dim) -> np.ndarray:
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        while not done:
            action = actor.act(state, device)
            action = action + np.random.normal(0, explore_noise, size=action_dim)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards)


def return_reward_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def modify_reward(dataset, env_name, reward_scale=None, reward_bias=None, max_episode_steps=1000):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        if reward_scale is not None and reward_bias is not None:
            dataset["rewards"] = dataset["rewards"] * reward_scale + reward_bias
        else:
            dataset["rewards"] -= 1.0


def modify_reward_online(env_name, reward, reward_scale=None, reward_bias=None):
    assert "antmaze" in env_name
    if reward_scale is not None and reward_bias is not None:
        reward = reward * reward_scale + reward_bias
    else:
        reward -= 1.0
    return reward


class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape, mean=None, std=None):  # shape:the dimension of input data
        if mean is not None:
            self.mean = mean
            self.std = std
            self.S = std ** 2
        else:
            self.mean = np.zeros(shape)
            self.S = np.ones(shape)
            self.std = np.sqrt(self.S)
        self.n = 0

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)


class Normalization:
    def __init__(self, shape, mean=None, std=None):
        self.running_ms = RunningMeanStd(shape=shape, mean=mean, std=std)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=False
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std)  # + 1e-8)

        return x


class RewardScaling:
    def __init__(self, shape, gamma, scaling='none', env='hopper-medium-v2'):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)
        self.type = scaling
        self.env = env

    def __call__(self, x):
        if self.type == 'scaling':
            self.R = self.gamma * self.R + x
            self.running_ms.update(self.R)
            x = x / (self.running_ms.std + 1e-8)  # Only divided std
            return x
        elif self.type == 'number':
            return 0.1 * x
        else:
            if "antmaze" in self.env:
                return x - 1.0
            else:
                return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)

def is_goal_reached(reward: float, info) -> bool:
    if "goal_achieved" in info:
        return info["goal_achieved"]
    return reward > 0


def load_train_config(file_path, config):
    with open(file_path, 'r') as file:
        config_data = yaml.safe_load(file)
        config_fields = fields(config)

        filtered_config_data = {field.name: config_data[field.name] for field in config_fields if
                                field.name in config_data}
        config = replace(config, **filtered_config_data)
        return config


def load_train_config_auto(config, stage, method):
    env_higher = "_".join(config.env.split("-")[:1]).lower().replace("-", "_")
    env_lower = "_".join(config.env.split("-")[1:]).lower().replace("-", "_")

    config_file_path = os.path.join(f"../config/{stage}/{method}/{env_higher}", f"{env_lower}.yaml")

    return load_train_config(config_file_path, config)
