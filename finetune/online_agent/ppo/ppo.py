# source: https://github.com/Lei-Kun/Uni-O4/blob/master/ppo_finetune/ppo.py
import copy

import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from .ppo_net import ValueMLP, GaussianPolicy, ValueTanhMLP
import numpy as np
import gym


class PPO():
    def __init__(self, args, device):
        self.max_action = args.max_action
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr_a = args.lr_a  # Learning rate of actor
        self.lr_c = args.lr_c  # Learning rate of critic
        self.gamma = args.gamma  # Discount factor
        self.lamda = args.lamda  # GAE parameter
        self.epsilon = args.epsilon  # PPO clip parameter
        self.K_epochs = args.K_epochs  # PPO parameter
        self.entropy_coef = args.entropy_coef  # Entropy coefficient
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm
        self.device = device
        self.args = args
        self.set_critic_count = 0

        if args.is_decay_pi:
            self.update_actor = GaussianPolicy(args.state_dim, args.action_dim, args.max_action, ).to(self.device)
        self.actor = GaussianPolicy(args.state_dim, args.action_dim, args.max_action, ).to(self.device)
        self.critic = ValueTanhMLP(args).to(self.device)

        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            if args.is_decay_pi:
                self.optimizer_actor = torch.optim.Adam(self.update_actor.parameters(), lr=self.lr_a, eps=1e-5)
            else:
                self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            if args.is_decay_pi:
                self.optimizer_actor = torch.optim.Adam(self.update_actor.parameters(), lr=self.lr_a)
            else:
                self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

        self.reference_actor = None
        self.alpha = args.alpha
        self.end_steps = args.end_steps
        self.use_auxi = args.use_auxi

    def load_actor(self, pi_path):
        actor_dict = torch.load(pi_path)['actor']
        self.actor.load_state_dict(actor_dict)
        self.reference_actor = copy.deepcopy(self.actor)
        print('Policy parameters loaded')

    def load_value(self, value_path: str):
        self.critic.load_state_dict(torch.load(value_path))
        print('Value parameters loaded')

    def save_value(self, value_path: str):
        torch.save(self.critic.state_dict(), value_path)

    def off_evaluate(
            self,
            env_name: str,
            seed: int,
            mean: np.ndarray,
            std: np.ndarray,
            eval_episodes: int = 10
    ) -> (float, float, float, float):
        env = gym.make(env_name)
        env.seed(seed)

        total_reward = 0
        values, first_values = [], []
        for _ in range(eval_episodes):
            s, done = env.reset(), False
            it = 0
            while not done:

                s = torch.FloatTensor((np.array(s).reshape(1, -1) - mean) / std).to(self.device)
                value = self.critic(s)
                values.append(value.cpu().data.numpy().flatten().tolist())
                if it == 0:
                    first_values.append(value.cpu().data.numpy().flatten().tolist())
                a = self.actor(s).cpu().data.numpy().flatten()
                s, r, done, info = env.step(a)

                total_reward += r
                it += 1

        avg_reward = total_reward / eval_episodes
        d4rl_score = env.get_normalized_score(avg_reward) * 100
        return avg_reward, d4rl_score, np.mean(values), np.mean(first_values)

    def evaluate(self, s):  # When evaluating the policy, we only use the mean
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(self.device)

        a = self.actor(s).detach().cpu().numpy().flatten()
        return a

    def choose_action(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(self.device)

        with torch.no_grad():
            dist = self.actor.get_dist(s)
            a = dist.sample()  # Sample the action according to the probability distribution
            a = torch.clamp(a, -self.max_action, self.max_action)  # [-max,max]
            a_logprob = dist.log_prob(a)  # The log probability density of the action
        return a.cpu().numpy().flatten(), a_logprob.cpu().numpy().flatten()

    def get_reference_advantage(self, s, a):
        old_dist = self.reference_actor.get_dist(s)
        entropy = old_dist.entropy().sum(-1)
        old_log_prob = old_dist.log_prob(a).sum(-1)
        adv_q = old_log_prob + entropy
        action_dim = a.shape[-1]
        adv_q = F.softplus(adv_q + 4 * action_dim) - 4 * action_dim

        return self.alpha * adv_q / action_dim

    def update(self, replay_buffer, total_steps):
        s, a, a_logprob, r, s_, dw, done = replay_buffer.numpy_to_tensor()  # Get training data

        adv = []
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            vs = self.critic(s)
            vs_ = self.critic(s_)
            deltas = r + self.gamma * (1.0 - dw) * vs_ - vs
            for delta, d in zip(reversed(deltas.flatten().cpu().numpy()), reversed(done.flatten().cpu().numpy())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1).to(self.device)
            v_target = adv + vs

            if self.use_auxi:
                time_weight = max(1.0 - total_steps / self.end_steps, 0)
                adv_ref = self.get_reference_advantage(s, a) * torch.abs(adv.std())
                adv = (adv + adv_ref * time_weight) / (1 + time_weight)

            if self.use_adv_norm:  # Trick 1:advantage normalization
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        # Optimize policy for K epochs:
        actor_losses, critic_losses = [], []
        for times in range(self.K_epochs):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                if self.args.is_decay_pi:
                    dist_now = self.update_actor.get_dist(s[index])
                    dist_entropy = dist_now.entropy().sum(1, keepdim=True)  # shape(mini_batch_size X 1)
                    a_logprob_now = dist_now.log_prob(a[index])
                    # a/b=exp(log(a)-log(b))  In multi-dimensional continuous action space，we need to sum up the log_prob
                    ratios = torch.exp(a_logprob_now.sum(1, keepdim=True) - a_logprob[index].sum(1,
                                                                                                 keepdim=True))  # shape(mini_batch_size X 1)

                    surr1 = ratios * adv[index]  # Only calculate the gradient of 'a_logprob_now' in ratios
                    surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                    actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # Trick 5: policy entropy
                    actor_losses.append(actor_loss.mean().item())
                    # Update actor
                    self.optimizer_actor.zero_grad()
                    actor_loss.mean().backward()
                    if self.use_grad_clip:  # Trick 7: Gradient clip
                        torch.nn.utils.clip_grad_norm_(self.update_actor.parameters(), 0.5)
                    self.optimizer_actor.step()
                    for param, target_param in zip(self.update_actor.parameters(), self.actor.parameters()):
                        target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
                else:
                    dist_now = self.actor.get_dist(s[index])
                    dist_entropy = dist_now.entropy().sum(1, keepdim=True)  # shape(mini_batch_size X 1)
                    a_logprob_now = dist_now.log_prob(a[index])
                    # a/b=exp(log(a)-log(b))  In multi-dimensional continuous action space，we need to sum up the log_prob
                    ratios = torch.exp(a_logprob_now.sum(1, keepdim=True) - a_logprob[index].sum(1,
                                                                                                 keepdim=True))  # shape(mini_batch_size X 1)

                    surr1 = ratios * adv[index]  # Only calculate the gradient of 'a_logprob_now' in ratios
                    surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                    actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # Trick 5: policy entropy
                    actor_losses.append(actor_loss.mean().item())
                    # Update actor
                    self.optimizer_actor.zero_grad()
                    actor_loss.mean().backward()
                    if self.use_grad_clip:  # Trick 7: Gradient clip
                        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                    self.optimizer_actor.step()

                v_s = self.critic(s[index])

                if self.args.is_clip_value:
                    old_value_clipped = vs[index] + (v_s - vs[index]).clamp(-self.epsilon, self.epsilon)
                    value_loss = (v_s - v_target[index].detach().float()).pow(2)
                    value_loss_clipped = (old_value_clipped - v_target[index].detach().float()).pow(2)
                    critic_loss = torch.max(value_loss, value_loss_clipped).mean()
                else:
                    critic_loss = F.mse_loss(v_target[index], v_s)
                critic_losses.append(critic_loss.mean().item())
                # Update critic
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

        if self.use_lr_decay:  # Trick 6:learning rate Decay
            self.lr_decay(total_steps)
        return np.mean(actor_losses), np.mean(critic_losses)

    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now

    def save(self, path):
        torch.save(self.actor.state_dict(), path + '/actor.pth')
        torch.save(self.critic.state_dict(), path + '/critic.pth')

    def ope(self, replay_buffer, batch_size):
        s, _, r, s_, _, not_done, _, _ = replay_buffer.sample(batch_size)
        with torch.no_grad():
            v_target = r + self.gamma * not_done * self.critic(s_)
        v_s = self.critic(s)
        value_loss = F.mse_loss(v_target, v_s)
        self.optimizer_critic.zero_grad()
        value_loss.backward()
        self.optimizer_critic.step()
        return value_loss.item()
