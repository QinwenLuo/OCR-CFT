# source: https://github.com/Lei-Kun/Uni-O4/blob/master/ppo_finetune/offline_buffer.py
import os
import torch
import numpy as np
from tqdm import tqdm
from .ppo_utils import CONST_EPS, RewardScaling, normalize
from copy import deepcopy


class OnlineReplayBuffer:
    _device: torch.device
    _state: np.ndarray
    _action: np.ndarray
    _reward: np.ndarray
    _next_state: np.ndarray
    _next_action: np.ndarray
    _not_done: np.ndarray
    _return: np.ndarray
    _size: int

    def __init__(
            self,
            device: torch.device,
            state_dim: int, action_dim: int, max_size: int, percentage: float
    ) -> None:
        self._percentage = percentage
        self._device = device

        self._state = np.zeros((max_size, state_dim))
        self._action = np.zeros((max_size, action_dim))
        self._reward = np.zeros((max_size, 1))
        self._next_state = np.zeros((max_size, state_dim))
        self._next_action = np.zeros((max_size, action_dim))
        self._not_done = np.zeros((max_size, 1))
        self._return = np.zeros((max_size, 1))
        self._advantage = np.zeros((max_size, 1))

        self._size = 0

    def store(
            self,
            s: np.ndarray,
            a: np.ndarray,
            r: np.ndarray,
            s_p: np.ndarray,
            a_p: np.ndarray,
            not_done: bool
    ) -> None:

        self._state[self._size] = s
        self._action[self._size] = a
        self._reward[self._size] = r
        self._next_state[self._size] = s_p
        self._next_action[self._size] = a_p
        self._not_done[self._size] = not_done
        self._size += 1

    def compute_return(
            self, gamma: float
    ) -> None:

        pre_return = 0
        for i in tqdm(reversed(range(self._size)), desc='Computing the returns'):
            self._return[i] = self._reward[i] + gamma * pre_return * self._not_done[i]
            pre_return = self._return[i]

    def compute_advantage(
            self, gamma: float, lamda: float, value
    ) -> None:
        delta = np.zeros_like(self._reward)

        pre_value = 0
        pre_advantage = 0

        for i in tqdm(reversed(range(self._size)), 'Computing the advantage'):
            current_state = torch.FloatTensor(self._state[i]).to(self._device)
            current_value = value(current_state).cpu().data.numpy().flatten()

            delta[i] = self._reward[i] + gamma * pre_value * self._not_done[i] - current_value
            self._advantage[i] = delta[i] + gamma * lamda * pre_advantage * self._not_done[i]

            pre_value = current_value
            pre_advantage = self._advantage[i]

        self._advantage = (self._advantage - self._advantage.mean()) / (self._advantage.std() + CONST_EPS)

    def shuffle(self, ):
        indices = np.arange(self._state.shape[0])
        np.random.shuffle(indices)
        self._state = self._state[indices]
        self._action = self._action[indices]
        self._reward = self._reward[indices]
        self._next_state = self._next_state[indices]
        self._next_action = self._next_action[indices]
        self._not_done = self._not_done[indices]
        self._return = self._return[indices]
        self._advantage = self._advantage[indices]

    def sample_all(self, ):

        return {
            "observations": self._state[:self._size].copy(),
            "actions": self._action[:self._size].copy(),
            "next_observations": self._next_state[:self._size].copy(),
            "terminals": 1. - self._not_done[:self._size].copy(),
            "rewards": self._reward[:self._size].copy()
        }

    def sample_aug_all(self, ):
        self._state = np.concatenate((self._state, self._aug_state), axis=0)
        self._action = np.concatenate((self._action, self._action), axis=0)
        self._next_state = np.concatenate((self._next_state, self._aug_next_state), axis=0)
        self._not_done = np.concatenate((self._not_done, self._not_done), axis=0)
        self._reward = np.concatenate((self._reward, self._reward), axis=0)
        indices = np.arange(self._state.shape[0])
        np.random.shuffle(indices)
        return {
            "observations": self._state[indices].copy(),
            "actions": self._action[indices].copy(),
            "next_observations": self._next_state[indices].copy(),
            "terminals": 1. - self._not_done[indices].copy(),
            "rewards": self._reward[indices].copy()
        }

    def sample(
            self, batch_size: int
    ) -> tuple:

        ind = np.random.randint(0, int(self._size * self._percentage), size=batch_size)

        return (
            torch.FloatTensor(self._state[ind]).to(self._device),
            torch.FloatTensor(self._action[ind]).to(self._device),
            torch.FloatTensor(self._reward[ind]).to(self._device),
            torch.FloatTensor(self._next_state[ind]).to(self._device),
            torch.FloatTensor(self._next_action[ind]).to(self._device),
            torch.FloatTensor(self._not_done[ind]).to(self._device),
            torch.FloatTensor(self._return[ind]).to(self._device),
            torch.FloatTensor(self._advantage[ind]).to(self._device)
        )

    def sample_aug_state(self, batch_size: int):
        self._state = np.concatenate((self._state, self._aug_state), axis=0)
        ind = np.random.randint(0, int(self._size * 2), size=batch_size)
        return (
            torch.FloatTensor(self._state[ind]).to(self._device)
        )

    def augmentaion(self, alpha=0.75, beta=1.25):
        z = np.random.uniform(low=alpha, high=beta, size=self._state.shape)
        self._aug_state = deepcopy(self._state) * z
        self._aug_next_state = deepcopy(self._next_state) * z

    def sample_percentage_eval(
            self, batch_size: int
    ) -> tuple:
        ind = np.random.randint(int(self._size * self._percentage), self._size, size=batch_size)

        return (
            torch.FloatTensor(self._state[ind]).to(self._device),
            torch.FloatTensor(self._action[ind]).to(self._device),
            torch.FloatTensor(self._reward[ind]).to(self._device),
            torch.FloatTensor(self._next_state[ind]).to(self._device),
            torch.FloatTensor(self._next_action[ind]).to(self._device),
            torch.FloatTensor(self._not_done[ind]).to(self._device),
            torch.FloatTensor(self._return[ind]).to(self._device),
            torch.FloatTensor(self._advantage[ind]).to(self._device)
        )


class OfflineReplayBuffer(OnlineReplayBuffer):

    def __init__(
            self, device: torch.device,
            state_dim: int, action_dim: int, max_size: int, percentage: float = 1.
    ) -> None:
        super().__init__(device, state_dim, action_dim, max_size, percentage)

    def load_dataset(
            self, dataset: dict, clip=False
    ) -> None:
        if clip:
            lim = 1. - 1e-5
            dataset['actions'] = np.clip(dataset['actions'], -lim, lim)
        self._state = dataset['observations'][:-1, :]

        self._action = dataset['actions'][:-1, :]
        self._reward = dataset['rewards'].reshape(-1, 1)[:-1, :]
        self._next_state = dataset['observations'][1:, :]
        self._next_action = dataset['actions'][1:, :]
        if 'timesout' in dataset.keys():
            self._not_done = 1. - (
                    dataset['terminals'].reshape(-1, 1)[:-1, :] | dataset['timeouts'].reshape(-1, 1)[:-1, :])
        else:
            self._not_done = 1. - dataset['terminals'].reshape(-1, 1)[:-1, :]
        self._size = len(dataset['actions']) - 1

    def reward_normalize(self, gamma=0.99, scaling='dynamic'):  # dynamic/normal/number
        if scaling == 'dynamic':
            print('scaling reward dynamically')
            reward_norm = RewardScaling(1, gamma)
            rewards = self._reward.flatten()
            for i, not_done in enumerate(self._not_done.flatten()):
                if not not_done:
                    reward_norm.reset()
                else:
                    rewards[i] = reward_norm(rewards[i])
            self._reward = rewards.reshape(-1, 1)

            return reward_norm
        elif scaling == 'normal':
            print('use normal reward scaling')
            normalized_rewards = normalize(self._state, self._action, deepcopy(self._reward.flatten()),
                                           self._not_done.flatten(), 1 - self._not_done.flatten(), self._next_state)
            self._reward = normalized_rewards.reshape(-1, 1)

        elif scaling == 'number':
            print('use a fixed number reward scaling')
            self._reward = self._reward * 0.1
        else:
            print('donnot use any reward scaling')
            self._reward = self._reward

    def normalize_state(
            self
    ) -> tuple:

        mean = self._state.mean(0, keepdims=True)
        std = self._state.std(0, keepdims=True) + CONST_EPS
        self._state = (self._state - mean) / std
        self._next_state = (self._next_state - mean) / std
        return (mean, std)
