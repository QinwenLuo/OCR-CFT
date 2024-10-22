from typing import Dict, List

import numpy as np
import torch
import copy

TensorBatch = List[torch.Tensor]


class ReplayBuffer:
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            buffer_size: int,
            device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, self._size, size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def add_transition(self,
                       state: np.ndarray,
                       action: np.ndarray,
                       reward: float,
                       next_state: np.ndarray,
                       done: bool
                       ):
        # Use this method to add new data into the replay buffer during fine-tuning.
        self._states[self._pointer] = self._to_tensor(state)
        self._actions[self._pointer] = self._to_tensor(action)
        self._rewards[self._pointer] = self._to_tensor(np.array(reward))
        self._next_states[self._pointer] = self._to_tensor(next_state)
        self._dones[self._pointer] = self._to_tensor(np.array(done))

        self._pointer = (self._pointer + 1) % self._buffer_size
        self._size = min(self._size + 1, self._buffer_size)

    def size(self):
        return self._size


class Online_ReplayBuffer:
    def __init__(self, state_dim, action_dim, buffer_size_size, device):
        self.online_buffer = ReplayBuffer(state_dim, action_dim, buffer_size_size, device)
        self.offline_buffer = None
        self.symbol = 0

    def initial(self, symbol: str, data: Dict[str, np.ndarray]):
        if symbol == 'all':
            self.online_buffer.load_d4rl_dataset(data)
        elif symbol == 'half':
            self.offline_buffer = copy.deepcopy(self.online_buffer)
            self.offline_buffer.load_d4rl_dataset(data)
            self.symbol = 1
        elif symbol == 'part':
            load_trajs(50, data, self.online_buffer)
        elif symbol == 'none':
            pass
        else:
            raise NameError

    def add_transition(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool):
        self.online_buffer.add_transition(state, action, reward, next_state, done)

    def sample(self, batch_size: int):
        if self.symbol == 1:
            online_batch = self.online_buffer.sample(batch_size // 2)
            offline_batch = self.offline_buffer.sample(batch_size // 2)
            batch = []
            for i in range(len(online_batch)):
                batch.append(torch.cat([online_batch[i], offline_batch[i]], axis=0))
        else:
            batch = self.online_buffer.sample(batch_size)
        return batch

    def size(self):
        if self.offline_buffer is not None:
            return self.online_buffer.size() + self.offline_buffer.size()
        else:
            return self.online_buffer.size()


class PPO_ReplayBuffer:
    def __init__(self, state_dim, action_dim, batch_size, device):
        self.s = np.zeros((batch_size, state_dim))
        self.a = np.zeros((batch_size, action_dim))
        self.a_logprob = np.zeros((batch_size, 1))
        self.r = np.zeros((batch_size, 1))
        self.s_ = np.zeros((batch_size, state_dim))
        self.dw = np.zeros((batch_size, 1))
        self.done = np.zeros((batch_size, 1))
        self.count = 0
        self.device = device

    def add_transition(self, s, a, a_logprob, r, s_, dw, done):
        self.s[self.count] = s
        self.a[self.count] = a
        self.a_logprob[self.count] = a_logprob
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.done[self.count] = done
        self.count += 1

    def numpy_to_tensor(self):
        s = torch.tensor(self.s, dtype=torch.float).to(self.device)
        a = torch.tensor(self.a, dtype=torch.float).to(self.device)
        a_logprob = torch.tensor(self.a_logprob, dtype=torch.float).to(self.device)
        r = torch.tensor(self.r, dtype=torch.float).to(self.device)
        s_ = torch.tensor(self.s_, dtype=torch.float).to(self.device)
        dw = torch.tensor(self.dw, dtype=torch.float).to(self.device)
        done = torch.tensor(self.done, dtype=torch.float).to(self.device)

        return s, a, a_logprob, r, s_, dw, done


class Return_ReplayBuffer(ReplayBuffer):
    def __init__(self, state_dim: int, action_dim: int, buffer_size: int, device: str = "cpu"):
        super().__init__(state_dim, action_dim, buffer_size, device)
        self._returns = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, self._size, size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        returns = self._returns[indices]
        return [states, actions, rewards, next_states, dones, returns]

    def compute_return(self, gamma: float):
        pre_return = 0
        for i in reversed(range(self._size)):
            self._returns[i] = self._rewards[i] + gamma * pre_return * (1 - self._dones[i])
            pre_return = self._returns[i]


def clip_dateset(dataset, ind):
    new_dataset = dict()
    new_dataset["observations"] = dataset["observations"][ind]
    new_dataset["actions"] = dataset["actions"][ind]
    new_dataset["rewards"] = dataset["rewards"][ind]
    new_dataset["next_observations"] = dataset["next_observations"][ind]
    new_dataset["terminals"] = dataset["terminals"][ind]
    return new_dataset


def load_trajs(nums, dataset, buffer):
    trajectories = []
    current_trajectory = {'observations': [], 'actions': [], 'rewards': [], 'next_observations': [], 'terminals': []}

    for i in range(len(dataset['terminals'])):
        current_trajectory['observations'].append(dataset['observations'][i])
        current_trajectory['actions'].append(dataset['actions'][i])
        current_trajectory['rewards'].append(dataset['rewards'][i])
        current_trajectory['next_observations'].append(dataset['next_observations'][i])
        current_trajectory['terminals'].append(dataset['terminals'][i])

        if i == len(dataset['terminals']) - 1:
            trajectories.append(current_trajectory)
        else:
            if np.linalg.norm(dataset['observations'][i + 1] -
                              dataset['next_observations'][i]) > 1e-6 or dataset['terminals'][i] == 1.0:
                trajectories.append(current_trajectory)
                current_trajectory = {'observations': [], 'actions': [], 'rewards': [], 'next_observations': [],
                                      'terminals': []}

    returns = [np.sum(trajectory['rewards']) for trajectory in trajectories]
    indices = np.argsort(returns)[::-1][:nums]
    selected_trajectories = [trajectories[i] for i in indices]
    new_dataset = {
        'observations': np.concatenate([np.array(trajectory['observations']) for trajectory in selected_trajectories]),
        'actions': np.concatenate([np.array(trajectory['actions']) for trajectory in selected_trajectories]),
        'rewards': np.concatenate([np.array(trajectory['rewards']) for trajectory in selected_trajectories]),
        'next_observations': np.concatenate(
            [np.array(trajectory['next_observations']) for trajectory in selected_trajectories]),
        'terminals': np.concatenate([np.array(trajectory['terminals']) for trajectory in selected_trajectories])
    }
    buffer.load_d4rl_dataset(new_dataset)
