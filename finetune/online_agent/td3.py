import sys
import copy

sys.path.append('...')

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any, Dict, List
from net import Scalar

TensorBatch = List[torch.Tensor]


def euclidean_distance(mean_actions, noisy_actions):
    action_dim = mean_actions.shape[-1]
    return (torch.linalg.norm((noisy_actions - mean_actions), dim=-1, keepdim=True)) / pow(action_dim, 0.5)


def asymmetric_reverse_l2_loss(u: torch.Tensor, tau: float = 0.7) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u > 0).float()) * u)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


class TD3:
    def __init__(
            self,
            max_action: float,
            actor: nn.Module,
            actor_optimizer: torch.optim.Optimizer,
            critic_1: nn.Module,
            critic_1_optimizer: torch.optim.Optimizer,
            critic_2: nn.Module,
            critic_2_optimizer: torch.optim.Optimizer,
            discount: float = 0.99,
            tau: float = 0.005,
            policy_noise: float = 0.2,
            noise_clip: float = 0.5,
            policy_freq: int = 2,
            beta: float = 1.0,
            log_lmbda: float = 1.0,
            threshold: float = 0.01,
            end_threshold: float = 0.1,
            max_timesteps: int = int(1e6),
            loss_tau: float = 0.7,
            device: str = "cpu"
    ):
        self.actor = actor
        self.actor_target = copy.deepcopy(actor)
        self.offline_actor = copy.deepcopy(actor)
        self.actor_optimizer = actor_optimizer
        self.critic_1 = critic_1
        self.critic_1_target = copy.deepcopy(critic_1)
        self.critic_1_optimizer = critic_1_optimizer
        self.critic_2 = critic_2
        self.critic_2_target = copy.deepcopy(critic_2)
        self.critic_2_optimizer = critic_2_optimizer

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.beta = beta
        self.refine_noise = self.policy_noise

        self.log_lmbda = Scalar(log_lmbda)
        self.log_lmbda_optimizer = torch.optim.Adam(self.log_lmbda.parameters(), lr=5e-5)

        self.eps = threshold
        self.end_eps = end_threshold
        self.max_timesteps = max_timesteps
        self.slope = (self.end_eps - self.eps) / self.max_timesteps
        self.loss_tau = loss_tau

        self.total_it = 0
        self.device = device

    def choose_action(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(self.device)
        with torch.no_grad():
            actions = self.actor(s)
        return actions.detach().cpu().numpy().flatten()

    def update(self, batch, total_steps, grad_clip=True) -> Dict[str, float]:
        log_dict = {}
        self.total_it += 1

        state, action, reward, next_state, done = batch
        not_done = 1 - done

        with torch.no_grad():
            # Select action according to actor and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )

            offline_next_action = self.offline_actor(next_state)
            online_next_action = self.actor(next_state)

            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )

            lmbda = self.log_lmbda().exp().detach()

            # Compute the target Q value
            target_q1 = self.critic_1_target(next_state, next_action)
            target_q2 = self.critic_2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - lmbda * F.mse_loss(online_next_action, offline_next_action)
            target_q = (reward + not_done * self.discount * target_q.unsqueeze(1)).squeeze(1)

        # Get current Q estimates
        current_q1 = self.critic_1(state, action)
        current_q2 = self.critic_2(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # Optimize the critic
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # Delayed actor updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            with torch.no_grad():
                offline_action = self.offline_actor(state)
                online_action = self.actor(state)

            new_eps = min(self.end_eps, self.slope * total_steps + self.eps)

            lmbda_loss_weight = asymmetric_reverse_l2_loss(new_eps - (offline_action - online_action) ** 2,
                                                           self.loss_tau).detach()
            lmbda_loss = self.log_lmbda().exp() * lmbda_loss_weight

            policy_action = self.actor(state)

            actor_q = self.critic_1(state, policy_action)

            q_weight = 1.0 / actor_q.abs().mean().detach()
            actor_loss = - q_weight * actor_q.mean() + lmbda * F.mse_loss(offline_action, policy_action)

            # Optimize the lmbda
            self.log_lmbda_optimizer.zero_grad()
            lmbda_loss.backward()
            self.log_lmbda_optimizer.step()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.25)
            self.actor_optimizer.step()

            # Update the frozen target models
            soft_update(self.critic_1_target, self.critic_1, self.tau)
            soft_update(self.critic_2_target, self.critic_2, self.tau)
            soft_update(self.actor_target, self.actor, self.tau)

            log_dict.update(
                dict(
                    actor_q=actor_q.mean().item(),
                    lmbda=lmbda,
                )
            )

        return log_dict

    def ope(self, batch) -> Dict[str, float]:
        state, action, reward, next_state, done = batch
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )

            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )

        not_done = 1 - done

        with torch.no_grad():
            target_q1 = self.critic_1_target(next_state, next_action)
            target_q2 = self.critic_2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = (reward + not_done * self.discount * target_q.unsqueeze(1)).squeeze(1)

        current_q1 = self.critic_1(state, action)
        current_q2 = self.critic_2(state, action)

        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        soft_update(self.critic_1_target, self.critic_1, self.tau)
        soft_update(self.critic_2_target, self.critic_2, self.tau)

        ope_log_dict = dict(ope_critic_loss=critic_loss.item(),
                            ope_averge_q=current_q1.mean().item(),
                            ope_target_q=target_q.mean().item(),
                            )

        return ope_log_dict

    def align_value(self, batch):
        self.total_it += 1

        state, actions, _, _, _ = batch

        with torch.no_grad():
            offline_action = self.offline_actor(state)
            online_action = self.actor(state)

            offline_action_target_q1 = self.critic_1_target(state, offline_action)
            offline_action_target_q2 = self.critic_2_target(state, offline_action)
            offline_action_target_q = torch.min(offline_action_target_q1, offline_action_target_q2)

            refine_noise = (torch.randn_like(online_action) * self.refine_noise).clamp(
                -self.noise_clip, self.noise_clip
            )

            noisy_action = (online_action + refine_noise).clamp(-self.max_action, self.max_action)

            distance = euclidean_distance(offline_action, noisy_action).squeeze(1).clamp(max=self.refine_noise)

            noisy_action_q1 = self.critic_1_target(state, noisy_action)
            noisy_action_q2 = self.critic_2_target(state, noisy_action)
            refine_target_q = torch.where(offline_action_target_q < 0,
                                          offline_action_target_q * (1 + self.beta * torch.square(distance)),
                                          offline_action_target_q * (1 / (1 + self.beta * torch.square(distance))))
            refine_target_q1 = torch.min(refine_target_q, noisy_action_q1)
            refine_target_q2 = torch.min(refine_target_q, noisy_action_q2)

        current_noisy_q1 = self.critic_1(state, noisy_action)
        current_noisy_q2 = self.critic_2(state, noisy_action)

        refine_loss = F.mse_loss(current_noisy_q1, refine_target_q1) + F.mse_loss(current_noisy_q2, refine_target_q2)

        current_policy_q1 = self.critic_1(state, offline_action)
        current_policy_q2 = self.critic_2(state, offline_action)

        fixed_loss = F.mse_loss(current_policy_q1, offline_action_target_q) + F.mse_loss(current_policy_q2,
                                                                                         offline_action_target_q)

        critic_loss = refine_loss + fixed_loss

        actor_q = self.critic_1(state, self.actor(state))
        actor_loss = -actor_q.mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        log_dict = dict(refine_loss=refine_loss.item(),
                        refine_fixed_loss=fixed_loss.item(),
                        refine_total_loss=critic_loss.item(),
                        refine_target_q1=refine_target_q1.mean(),
                        refine_current_noisy_q1=current_noisy_q1.mean(),
                        refine_current_policy_q1=current_policy_q1.mean(),
                        refine_distance=distance.mean(),
                        reconstruct_actor_loss=actor_loss.item(),
                        reconstruct_actor_q=actor_q.mean()
                        )

        return log_dict

    def load_offline(self, state_dict):
        self.offline_actor.load_state_dict(state_dict=state_dict["actor"])
        self.actor.load_state_dict(state_dict=state_dict["actor"])
        self.actor_target = copy.deepcopy(self.actor)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "critic_1": self.critic_1.state_dict(),
            "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "total_it": self.total_it
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.critic_1.load_state_dict(state_dict["critic_1"])
        self.critic_1_optimizer.load_state_dict(state_dict["critic_1_optimizer"])
        self.critic_1_target = copy.deepcopy(self.critic_1)

        self.critic_2.load_state_dict(state_dict["critic_2"])
        self.critic_2_optimizer.load_state_dict(state_dict["critic_2_optimizer"])
        self.critic_2_target = copy.deepcopy(self.critic_2)

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_target = copy.deepcopy(self.actor)

        self.total_it = state_dict["total_it"]
