import sys
import copy

sys.path.append('...')

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, List
from net import Scalar

TensorBatch = List[torch.Tensor]
eps = 1e-5


def asymmetric_reverse_l2_loss(u: torch.Tensor, tau: float = 0.7) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u > 0).float()) * u)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


class SAC:
    def __init__(self,
                 critic_1,
                 critic_1_optimizer,
                 critic_2,
                 critic_2_optimizer,
                 actor,
                 actor_optimizer,
                 target_entropy: float,
                 max_action: float,
                 discount: float = 0.99,
                 alpha_multiplier: float = 1.0,
                 alpha_lr: float = 3e-4,
                 target_update_period: int = 1,
                 soft_target_update_rate: float = 5e-3,
                 use_automatic_entropy_tuning: bool = True,
                 log_alpha: float = 0.0,
                 threshold: float = 0.01,
                 end_threshold: float = 0.1,
                 max_timesteps: int = int(1e6),
                 double_critic: bool = True,
                 log_lmbda: float = 1.0,
                 loss_tau: float = 0.7,
                 device: str = "cpu"
                 ):
        self.critic_1 = critic_1
        self.critic_2 = critic_2

        self.target_critic_1 = copy.deepcopy(self.critic_1).to(device)
        self.target_critic_2 = copy.deepcopy(self.critic_2).to(device)

        self.actor = actor
        self.offline_actor = copy.deepcopy(self.actor).to(device)

        self.actor_optimizer = actor_optimizer
        self.critic_1_optimizer = critic_1_optimizer
        self.critic_2_optimizer = critic_2_optimizer

        self.discount = discount
        self.target_entropy = target_entropy
        self.alpha_multiplier = alpha_multiplier
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        self.max_action = max_action

        self.alpha_lr = alpha_lr
        self.target_update_period = target_update_period
        self.soft_target_update_rate = soft_target_update_rate

        self.eps = threshold
        self.end_eps = end_threshold
        self.max_timesteps = max_timesteps
        self.slope = (self.end_eps - self.eps) / self.max_timesteps

        self.loss_tau = loss_tau

        self.device = device

        if self.use_automatic_entropy_tuning:
            self.log_alpha = Scalar(log_alpha)
            self.alpha_optimizer = torch.optim.Adam(
                self.log_alpha.parameters(),
                lr=self.alpha_lr,
            )
        else:
            self.log_alpha = None

        self.log_lmbda = Scalar(log_lmbda)
        self.log_lmbda_optimizer = torch.optim.Adam(self.log_lmbda.parameters(), lr=5e-5)

        self.total_it = 0

        self.double_critic = double_critic

    def min_q_values(self, critic_1, critic_2, observations, actions):
        if self.double_critic:
            q1 = critic_1(observations, actions)
            q2 = critic_2(observations, actions)
            return torch.min(q1, q2)
        else:
            return critic_1(observations, actions)

    def compute_q_mse_loss(self, observations, actions, target_q):
        q1 = self.critic_1(observations, actions)
        qf1_loss = F.mse_loss(q1, target_q.detach())
        if self.double_critic:
            q2 = self.critic_2(observations, actions)
            qf2_loss = F.mse_loss(q2, target_q.detach())
            qf_loss = qf1_loss + qf2_loss
        else:
            qf_loss = qf1_loss
        return qf_loss, q1

    def choose_action(self, s, mode='online'):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(self.device)
        with torch.no_grad():
            if mode == 'offline':
                dist = self.offline_actor.get_dist(s)
            else:
                dist = self.actor.get_dist(s)
            a = dist.sample()
            a = torch.clamp(a, -self.max_action, self.max_action)

        return a.detach().cpu().numpy().flatten()

    def update_target_network(self, soft_target_update_rate: float):
        soft_update(self.target_critic_1, self.critic_1, soft_target_update_rate)
        soft_update(self.target_critic_2, self.critic_2, soft_target_update_rate)

    def _alpha_and_alpha_loss(self, observations: torch.Tensor, log_pi: torch.Tensor):
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(
                    self.log_alpha() * (log_pi + self.target_entropy).detach()
            ).mean()
            alpha = self.log_alpha().exp() * self.alpha_multiplier
        else:
            alpha_loss = observations.new_tensor(0.0)
            alpha = observations.new_tensor(self.alpha_multiplier)
        return alpha, alpha_loss

    def _policy_loss(
            self,
            observations: torch.Tensor,
            new_actions: torch.Tensor,
            alpha: torch.Tensor,
            lmbda: torch.Tensor,
            log_pi: torch.Tensor,
            offline_log_pi: torch.Tensor,
    ) -> torch.Tensor:
        q_new_actions = self.min_q_values(self.critic_1, self.critic_2, observations, new_actions)

        policy_loss = (alpha * log_pi - q_new_actions + lmbda * (log_pi - offline_log_pi)).mean()

        return policy_loss

    def _q_loss(self,
                batch: TensorBatch,
                alpha: torch.Tensor,
                lmbda: torch.Tensor,
                log_dict: Dict,
                ) -> torch.Tensor:
        observations, actions, rewards, next_observations, dones = batch

        with torch.no_grad():
            new_next_actions, next_log_pi = self.actor(next_observations)
            target_q_values = self.min_q_values(self.target_critic_1, self.target_critic_2, next_observations,
                                                new_next_actions)

            offline_next_log_pi = self.offline_actor.log_prob(next_observations, new_next_actions).clamp(min=-50)

            target_q_values = target_q_values - alpha * next_log_pi - lmbda * (next_log_pi - offline_next_log_pi)

            td_target = (rewards + (1.0 - dones) * self.discount * target_q_values.unsqueeze(1)).squeeze(1)

        qf_loss, q1_predicted = self.compute_q_mse_loss(observations, actions, td_target)

        log_dict.update(
            dict(
                qfloss=qf_loss.item(),
                average_qf1=q1_predicted.mean().item(),
                td_target=td_target.mean().item()
            )
        )

        return qf_loss

    def _lmbda_loss(
            self,
            log_pi: torch.Tensor,
            offline_log_pi: torch.Tensor,
            total_steps: int,
    ):
        new_eps = min(self.end_eps, self.slope * total_steps + self.eps)

        lmbda_loss_weight = asymmetric_reverse_l2_loss(new_eps - log_pi + offline_log_pi, self.loss_tau).detach()
        lmbda_loss = self.log_lmbda().exp() * lmbda_loss_weight

        lmbda = self.log_lmbda().exp()

        return lmbda, lmbda_loss

    def update(self, batch, total_steps, grad_clip=False, times=0) -> Dict[str, float]:
        observations, actions, rewards, next_observations, dones = batch

        self.total_it += 1

        new_actions, log_pi = self.actor(observations)

        offline_log_pi = self.offline_actor.log_prob(observations, new_actions).clamp(min=-50)

        alpha, alpha_loss = self._alpha_and_alpha_loss(observations, log_pi)

        """ lmbda loss """
        if times == 0:
            lmbda, lmbda_loss = self._lmbda_loss(log_pi, offline_log_pi, total_steps)
        else:
            lmbda = self.log_lmbda().exp()

        """ Policy loss """
        policy_loss = self._policy_loss(observations, new_actions, alpha, lmbda, log_pi, offline_log_pi)

        log_dict = dict(
            alpha=alpha.item(),
            lmbda=lmbda.item(),
        )

        """ Q function loss """
        qf_loss = self._q_loss(batch, alpha, lmbda, log_dict)

        if times == 0:
            self.log_lmbda_optimizer.zero_grad()
            lmbda_loss.backward()
            self.log_lmbda_optimizer.step()

        if self.use_automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.25)
        self.actor_optimizer.step()

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        qf_loss.backward(retain_graph=True)
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        if self.total_it % self.target_update_period == 0:
            self.update_target_network(self.soft_target_update_rate)

        return log_dict

    def ope(self, batch: TensorBatch) -> Dict[str, float]:
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        ) = batch

        self.total_it += 1

        log_dict = dict()

        alpha = self.log_alpha().exp() * self.alpha_multiplier

        with torch.no_grad():
            new_next_actions, next_log_pi = self.offline_actor(next_observations)
            target_q_values = self.min_q_values(self.target_critic_1, self.target_critic_2, next_observations,
                                                new_next_actions)
            target_q_values = target_q_values - alpha * next_log_pi

            td_target = (rewards + (1.0 - dones) * self.discount * target_q_values.unsqueeze(1)).squeeze(1)

        qf_loss, q1_predicted = self.compute_q_mse_loss(observations, actions, td_target)

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        qf_loss.backward(retain_graph=True)
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        if self.total_it % self.target_update_period == 0:
            self.update_target_network(self.soft_target_update_rate)

        log_dict.update(
            dict(
                ope_average_q1=q1_predicted.mean().item(),
                ope_target_q=td_target.mean().item(),
                ope_loss=qf_loss.item()
            )
        )

        return log_dict

    def align_value(self, batch: TensorBatch) -> Dict[str, float]:
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        ) = batch

        self.total_it += 1

        alpha = self.log_alpha().exp() * self.alpha_multiplier

        with torch.no_grad():
            max_actions, max_log_probs = self.offline_actor(observations, deterministic=True)
            max_q_values = self.min_q_values(self.target_critic_1, self.target_critic_2, observations, max_actions)
            new_actions, _ = self.actor(observations)
            new_log_probs = self.offline_actor.log_prob(observations, new_actions)
            target_q_values = max_q_values - alpha * (max_log_probs - new_log_probs)

            old_q_values = self.min_q_values(self.target_critic_1, self.target_critic_2, observations, new_actions)

            target_q_values = torch.min(target_q_values, old_q_values)

        fixed_loss, max_q1_predicted = self.compute_q_mse_loss(observations, max_actions, max_q_values)

        predicted_loss, q1_predicted = self.compute_q_mse_loss(observations, new_actions, target_q_values)

        qf_loss = predicted_loss + fixed_loss

        new_actions, new_log_probs = self.actor(observations)

        q_new_actions = self.min_q_values(self.critic_1, self.critic_2, observations, new_actions)

        policy_loss = (alpha * new_log_probs - q_new_actions).mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        qf_loss.backward(retain_graph=True)
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        log_dict = dict(
            refine_average_qf1=q1_predicted.mean().item(),
            refine_average_target_q=target_q_values.mean().item(),
            refine_average_max_q1=max_q1_predicted.mean().item(),
            refine_predicted_loss=predicted_loss.item(),
            refine_fixed_loss=fixed_loss.item(),
            refine_qf_loss=qf_loss.item(),
            reconstruct_log_pi=new_log_probs.mean().item(),
            reconstruct_policy_loss=policy_loss.item()
        )

        return log_dict

    def load_offline(self, state_dict):
        self.offline_actor.load_state_dict(state_dict=state_dict["actor"])
        self.actor.load_state_dict(state_dict=state_dict["actor"])

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "critic1": self.critic_1.state_dict(),
            "critic2": self.critic_2.state_dict(),
            "critic1_target": self.target_critic_1.state_dict(),
            "critic2_target": self.target_critic_2.state_dict(),
            "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
            "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
            "actor_optim": self.actor_optimizer.state_dict(),
            "sac_log_alpha": self.log_alpha.state_dict(),
            "sac_log_alpha_optim": self.alpha_optimizer.state_dict()
        }

    def load_state_dict(self, state_dict):
        self.actor.load_state_dict(state_dict=state_dict["actor"])
        self.critic_1.load_state_dict(state_dict=state_dict["critic1"])
        self.critic_2.load_state_dict(state_dict=state_dict["critic2"])

        self.target_critic_1.load_state_dict(state_dict=state_dict["critic1_target"])
        self.target_critic_2.load_state_dict(state_dict=state_dict["critic2_target"])

        self.critic_1_optimizer.load_state_dict(
            state_dict=state_dict["critic_1_optimizer"]
        )
        self.critic_2_optimizer.load_state_dict(
            state_dict=state_dict["critic_2_optimizer"]
        )
        self.actor_optimizer.load_state_dict(state_dict=state_dict["actor_optim"])

        self.log_alpha.load_state_dict(state_dict["sac_log_alpha"])
        self.alpha_optimizer.load_state_dict(
            state_dict=state_dict["sac_log_alpha_optim"]
        )
