import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, Callable
from torch.distributions import Normal, TanhTransform, TransformedDistribution

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> nn.Parameter:
        return self.constant


def extend_and_repeat(tensor: torch.Tensor, dim: int, repeat: int) -> torch.Tensor:
    return tensor.unsqueeze(dim).repeat_interleave(repeat, dim=dim)


def orthogonal_net(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


def init_module_weights(module: torch.nn.Sequential, orthogonal_init: bool = False):
    # Specific orthogonal initialization for inner layers
    # If orthogonal init is off, we do not change default initialization
    if orthogonal_init:
        for submodule in module[:-1]:
            if isinstance(submodule, nn.Linear):
                nn.init.orthogonal_(submodule.weight, gain=np.sqrt(2))
                nn.init.constant_(submodule.bias, 0.0)

    # Lasy layers should be initialized differently as well
    if isinstance(module[-1], nn.Linear):
        if orthogonal_init:
            nn.init.orthogonal_(module[-1].weight, gain=1e-2)
        else:
            nn.init.xavier_uniform_(module[-1].weight, gain=1e-2)

        nn.init.constant_(module[-1].bias, 0.0)


class ReparameterizedTanhGaussian(nn.Module):
    def __init__(
            self, log_std_min: float = -20.0, log_std_max: float = 2.0, no_tanh: bool = False
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.no_tanh = no_tanh

    def log_prob(
            self, mean: torch.Tensor, log_std: torch.Tensor, sample: torch.Tensor
    ) -> torch.Tensor:
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        if self.no_tanh:
            action_distribution = Normal(torch.tanh(mean), std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )
        return torch.sum(action_distribution.log_prob(sample), dim=-1)

    def forward(
            self, mean: torch.Tensor, log_std: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        if self.no_tanh:
            action_distribution = Normal(torch.tanh(mean), std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )

        if deterministic:
            action_sample = torch.tanh(mean)
            if not self.no_tanh:
                action_sample = torch.clamp(action_sample, -0.9999, 0.9999)
        else:
            action_sample = action_distribution.rsample()

        log_prob = torch.sum(action_distribution.log_prob(action_sample), dim=-1)

        return action_sample, log_prob

    def get_dist(self, mean: torch.Tensor, log_std: torch.Tensor):
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        if self.no_tanh:
            mu = torch.tanh(mean)
            action_distribution = Normal(mu, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )
        return action_distribution


class TanhGaussianPolicy(nn.Module):
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            max_action: float,
            log_std_multiplier: float = 1.0,
            log_std_offset: float = -1.0,
            orthogonal_init: bool = False,
            no_tanh: bool = False,
    ):
        super().__init__()
        self.observation_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.orthogonal_init = orthogonal_init
        self.no_tanh = no_tanh

        self.base_network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * action_dim),
        )

        init_module_weights(self.base_network)

        self.log_std_multiplier = Scalar(log_std_multiplier)
        self.log_std_offset = Scalar(log_std_offset)
        self.tanh_gaussian = ReparameterizedTanhGaussian(no_tanh=no_tanh)

    def log_prob(
            self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        if actions.ndim == 3:
            observations = extend_and_repeat(observations, 1, actions.shape[1])
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        actions = torch.clamp(actions, -self.max_action+0.0001, self.max_action-0.0001)
        return self.tanh_gaussian.log_prob(mean, log_std, actions)

    def forward(
            self,
            observations: torch.Tensor,
            deterministic: bool = False,
            repeat: bool = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        actions, log_probs = self.tanh_gaussian(mean, log_std, deterministic)
        return self.max_action * actions, log_probs

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        with torch.no_grad():
            actions, _ = self(state, not self.training)
        return actions.cpu().data.numpy().flatten()

    @torch.no_grad()
    def log_prob_away_from_mean(self, observations: torch.Tensor, n=1):
        assert self.no_tanh == True
        with torch.no_grad():
            base_network_output = self.base_network(observations)
            mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
            log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
            log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
            std = torch.exp(log_std)
            action_1 = torch.tanh((mean + n * std)).clamp(-self.max_action+0.0001, self.max_action-0.0001)
            action_2 = torch.tanh((mean - n * std)).clamp(-self.max_action+0.0001, self.max_action-0.0001)
            log_prob = torch.max(self.tanh_gaussian.log_prob(mean, log_std, action_1),
                                 self.tanh_gaussian.log_prob(mean, log_std, action_2))
        return log_prob

    def get_dist(self, observations: torch.Tensor):
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        return self.tanh_gaussian.get_dist(mean, log_std)
    
    @torch.no_grad()
    def get_mean_std(self, observations: torch.Tensor):
        dist = self.get_dist(observations)
        return dist.mean, dist.scale


class GaussianPolicy(nn.Module):
    def __init__(
            self,
            state_dim: int,
            act_dim: int,
            max_action: float,
            hidden_dim: int = 256,
            n_hidden: int = 2,
            activation_fn=nn.ReLU,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            activation_fn=activation_fn,
            output_activation_fn=nn.Tanh,
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.max_action = max_action

    def reset_std(self, new_std):
        self.log_std.requires_grad = False
        self.log_std[self.log_std > new_std] = new_std
        self.log_std.requires_grad = True

    def get_dist(self, observations: torch.Tensor):
        mean = self.net(observations)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        return Normal(mean, std)

    def log_prob(
            self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        dist = self.get_dist(observations)
        return dist.log_prob(actions).sum(-1)

    def forward(self, observations: torch.Tensor):
        dist = self.get_dist(observations)
        actions = dist.rsample()
        log_probs = dist.log_prob(actions).sum(-1)
        return self.max_action * actions, log_probs

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        dist = self.get_dist(state)
        action = dist.mean if not self.training else dist.sample()
        action = torch.clamp(self.max_action * action, -self.max_action, self.max_action)
        return action.cpu().data.numpy().flatten()


class DeterministicPolicy(nn.Module):
    def __init__(
            self,
            state_dim: int,
            act_dim: int,
            max_action: float,
            hidden_dim: int = 256,
            n_hidden: int = 2,
            dropout: Optional[float] = None,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
            dropout=dropout,
        )
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return (
            torch.clamp(self(state) * self.max_action, -self.max_action, self.max_action)
            .cpu()
            .data.numpy()
            .flatten()
        )


class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


class MLP(nn.Module):
    def __init__(
            self,
            dims,
            activation_fn: Callable[[], nn.Module] = nn.ReLU,
            output_activation_fn: Callable[[], nn.Module] = None,
            squeeze_output: bool = False,
            dropout: Optional[float] = None,
            layer_norm: bool = False
    ):
        super().__init__()
        n_dims = len(dims)
        if n_dims < 2:
            raise ValueError("MLP requires at least two dims (input and output)")

        layers = []
        for i in range(n_dims - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LayerNorm(dims[i + 1]) if layer_norm else nn.Identity())
            layers.append(activation_fn())

            if dropout is not None:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
            layers.append(Squeeze(-1))
        for submodule in layers:
            if isinstance(submodule, nn.Linear):
                orthogonal_net(submodule)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CriticFunction(nn.Module):
    def __init__(
            self,
            observation_dim: int,
            action_dim: int,
            hidden_dim: int = 256,
            n_hidden: int = 2,
            squeeze_output: bool = True,
            layer_norm: bool = False
    ):
        super().__init__()
        dims = [observation_dim + action_dim, *([hidden_dim] * n_hidden), 1]

        self.network = MLP(dims, squeeze_output=squeeze_output, layer_norm=layer_norm)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        multiple_actions = False
        batch_size = observations.shape[0]
        if actions.ndim == 3 and observations.ndim == 2:
            multiple_actions = True
            observations = extend_and_repeat(observations, 1, actions.shape[1]).reshape(
                -1, observations.shape[-1]
            )
            actions = actions.reshape(-1, actions.shape[-1])
        input_tensor = torch.cat([observations, actions], dim=-1)
        q_values = self.network(input_tensor)
        if multiple_actions:
            q_values = q_values.reshape(batch_size, -1)
        return q_values


class ValueFunction(nn.Module):
    def __init__(
            self,
            observation_dim: int,
            hidden_dim: int = 256,
            n_hidden: int = 2,
            layer_norm: bool = False,
            activation_fn=nn.ReLU,
    ):
        super().__init__()
        dims = [observation_dim, *([hidden_dim] * n_hidden), 1]

        self.network = MLP(dims, squeeze_output=True, layer_norm=layer_norm, activation_fn=activation_fn)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        v_values = self.network(observations)
        return v_values


class TwinQ(nn.Module):
    def __init__(
            self,
            observation_dim: int,
            action_dim: int,
            hidden_dim: int = 256,
            n_hidden: int = 2,
    ):
        super().__init__()
        super().__init__()
        dims = [observation_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = MLP(dims, squeeze_output=True)
        self.q2 = MLP(dims, squeeze_output=True)

    def both(
            self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.both(state, action))
