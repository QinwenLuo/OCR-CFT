# source: https://github.com/Lei-Kun/Uni-O4/blob/master/ppo_finetune/net.py
import torch
import torch.nn as nn
from typing import Optional, Tuple, Callable
from .offline_buffer import OnlineReplayBuffer
import torch.nn.functional as F
from torch.distributions import Normal


def soft_clamp(
        x: torch.Tensor, bound: tuple
) -> torch.Tensor:
    low, high = bound
    x = torch.tanh(x)
    x = low + 0.5 * (high - low) * (x + 1)
    return x


# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


def MLP(
        input_dim: int,
        hidden_dim: int,
        depth: int,
        output_dim: int,
        activation: str = 'relu',
        final_activation: str = None
) -> torch.nn.modules.container.Sequential:
    if activation == 'tanh':
        act_f = nn.Tanh()
    elif activation == 'relu':
        act_f = nn.ReLU()

    layers = [nn.Linear(input_dim, hidden_dim), act_f]
    for _ in range(depth - 1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(act_f)

    layers.append(nn.Linear(hidden_dim, output_dim))
    if final_activation == 'relu':
        layers.append(nn.ReLU())
    elif final_activation == 'tanh':
        layers.append(nn.Tanh())
    else:
        layers = layers

    return nn.Sequential(*layers)


class ValueTanhMLP(nn.Module):
    _net: torch.nn.modules.container.Sequential

    def __init__(
            self, args
    ) -> None:
        super().__init__()
        self._net = MLP(args.state_dim, args.v_hidden_width, args.v_depth, 1, 'tanh')

    def forward(
            self, s: torch.Tensor
    ) -> torch.Tensor:
        return self._net(s)


class ValueMLP(nn.Module):
    _net: torch.nn.modules.container.Sequential

    def __init__(
            self, args
    ) -> None:
        super().__init__()
        self._net = MLP(args.state_dim, args.v_hidden_width, args.v_depth, 1, 'relu')

    def forward(
            self, s: torch.Tensor
    ) -> torch.Tensor:
        return self._net(s)


class GaussPolicyMLP(nn.Module):
    _net: torch.nn.modules.container.Sequential
    _log_std_bound: tuple

    def __init__(
            self, args
    ) -> None:
        super().__init__()
        if args.use_tanh:
            self._net = MLP(args.state_dim, args.hidden_width, args.depth, (2 * args.action_dim), 'tanh', 'tanh')
        else:
            self._net = MLP(args.state_dim, args.hidden_width, args.depth, (2 * args.action_dim), 'relu', 'tanh')
        self._log_std_bound = (-5., args.std_upper_bound)

    def forward(
            self, s: torch.Tensor
    ) -> torch.distributions:
        mu, _ = self._net(s).chunk(2, dim=-1)
        return mu

    def get_dist(
            self, s: torch.Tensor
    ) -> torch.distributions:

        mu, log_std = self._net(s).chunk(2, dim=-1)
        log_std = soft_clamp(log_std, self._log_std_bound)
        std = log_std.exp()

        dist = Normal(mu, std)
        return dist


class GaussPolicyMLP_(nn.Module):
    _net: torch.nn.modules.container.Sequential
    _log_std_bound: tuple

    def __init__(self, args,
                 scale_min=1e-4, scale_max=1., ):
        super(GaussPolicyMLP_, self).__init__()
        self.max_action = args.max_action
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.mean_layer = nn.Linear(args.hidden_width, 2 * args.action_dim)
        # self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))  # We use 'nn.Parameter' to train log_std automatically
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.mean_layer, gain=0.01)
        self._log_std_bound = (-5., 0.)

    def forward(
            self, s: torch.Tensor
    ) -> torch.distributions:
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        mu, _ = self.activate_func(self.mean_layer(s)).chunk(2, dim=-1)

        return mu

    def get_dist(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        mu, log_std = self.activate_func(self.mean_layer(s)).chunk(2, dim=-1)
        log_std = soft_clamp(log_std, self._log_std_bound)
        std = log_std.exp()

        dist = Normal(mu, std)
        return dist


class ValueLearner:
    _device: torch.device
    _value: ValueTanhMLP
    _optimizer: torch.optim
    _batch_size: int

    def __init__(
            self,
            args,
            value_lr: float,
            batch_size: int
    ) -> None:
        super().__init__()
        self._device = args.device
        self._value = ValueTanhMLP(args).to(args.device)
        self._optimizer = torch.optim.Adam(
            self._value.parameters(),
            lr=value_lr,
        )
        self._batch_size = batch_size

    def __call__(
            self, s: torch.Tensor
    ) -> torch.Tensor:
        return self._value(s)

    def update(
            self, replay_buffer: OnlineReplayBuffer
    ) -> float:
        s, _, _, _, _, _, Return, _ = replay_buffer.sample(self._batch_size)
        value_loss = F.mse_loss(self._value(s), Return)

        self._optimizer.zero_grad()
        value_loss.backward()
        self._optimizer.step()

        return value_loss.item()

    def save(
            self, path: str
    ) -> None:
        torch.save(self._value.state_dict(), path)
        print('Value parameters saved in {}'.format(path))

    def load(
            self, path: str
    ) -> None:
        self._value.load_state_dict(torch.load(path, map_location=self._device))
        print('Value parameters loaded')


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
        self.net = PolicyMLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            activation_fn=activation_fn,
            output_activation_fn=nn.Tanh,
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.max_action = max_action

    def forward(
            self,
            observations: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.net(observations)
        return mean

    def get_dist(self, observations: torch.Tensor):
        mean = self.net(observations)
        std = torch.exp(self.log_std.clamp(-10.0, 2.0))
        return Normal(mean, std)

    def reset_std(self, new_std):
        self.log_std.requires_grad = False
        self.log_std[self.log_std > new_std] = new_std
        self.log_std.requires_grad = True


class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


def orthogonal_net(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

    
class PolicyMLP(nn.Module):
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