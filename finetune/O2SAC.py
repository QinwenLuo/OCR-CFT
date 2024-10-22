import os
import sys
import copy
import tqdm
import json
import pyrallis

sys.path.append('..')

import torch
import numpy as np

import d4rl
import gym

from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from tensorboardX import SummaryWriter

from net import TanhGaussianPolicy, CriticFunction, Scalar
from dataset_utils import ReplayBuffer, Online_ReplayBuffer
from utils import modify_reward, compute_mean_std, normalize_states, wrap_env, set_seed, eval_actor, \
    eval_actor_explore, modify_reward_online, is_goal_reached, load_train_config_auto
from online_agent.sac import SAC

ENVS_WITH_GOAL = ("antmaze", "pen", "door", "hammer", "relocate")


@dataclass
class TrainConfig:
    # Experiment
    device: str = "cuda"
    env: str = "hopper-medium-v2"  # OpenAI gym environment name
    seed: int = 1  # Sets Gym, PyTorch and Numpy seeds
    eval_freq: int = int(1e3)  # How often (time steps) we evaluate
    n_episodes: int = 10  # How many episodes run during evaluation
    load_model: str = ""  # Model load file name, "" doesn't load

    # finetune initialing setting
    q_n_hidden_layers: int = 3  # Number of hidden layers in Q networks
    v_n_hidden_layers: int = 3  # Number of hidden layers in V networks
    alpha_multiplier: float = 1.0  # Multiplier for alpha in loss
    target_update_period: int = 1  # Frequency of target nets updates
    soft_target_update_rate: float = 5e-3  # Target network update rate
    use_automatic_entropy_tuning: bool = True  # Tune entropy
    buffer_size: int = 2_000_000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    policy_lr: float = 1e-4  # Actor learning rate
    alpha_lr: float = 3e-4  # Temperature factor learning rate
    vf_lr: float = 3e-4  # V function learning rate
    qf_lr: float = 3e-4  # Critic learning rate
    normalize: bool = True  # Normalize states
    normalize_reward: bool = False  # Normalize reward
    actor_dropout: Optional[float] = None  # Adroit uses dropout for policy network
    orthogonal_init: bool = True  # Orthogonal initialization
    policy_log_std_multiplier: float = 1.0
    reward_scale: float = 1.0
    reward_bias: float = 0.0

    # off2on setting
    actor_model: str = 'tanh'  # use squashed Gaussian distribution
    critic_layernorm: bool = True  # use layernorm trick for critic
    double_critic: bool = True  # use double critic trick
    log_freq: int = 5000  # How often (time steps) we log during optimistic critic reconstruction

    ope_steps: int = 500_000  # How many time steps run during policy re-evaluation
    align_steps: int = 500_000  # How many time steps run during value alignment
    warmup_steps: int = 1000  # update agent after warmup steps during online fine-tuning
    online_buffer_size: int = int(2e6)  # the size of replay buffer during online fine-tuning
    online_steps: int = int(1e5)  # the total interaction steps for online fine-tuning

    alpha: float = 0.2  # the pre-defined alpha used in policy re-evaluation
    sigma_KL_divergence: float = 0.5  # the pre-defined threshold of KL divergence for the actors
    threshold: float = sigma_KL_divergence / 4  # the initial threshold during constrained fine-tuning
    end_threshold: float = sigma_KL_divergence * 4  # the terminal threshold during constrained fine-tuning
    end_steps: int = int(1e5)  # the decay steps for the threshold during constrained fine-tuning

    lmbda: float = 1.0  # the initial value of lagrange multiplier in constrained fine-tuning
    loss_tau: float = 0.7  # the hyper-parameter in expectile regression

    buffer_load: str = 'half'  # how to use the offline data
    """
    all: load all offline data
    half: symmetric sample
    part: load part high-return offline data
    none: don't load any offline data
    """

    save_dir: str = './log'  # the directory saving the training log

    offline_alg: str = 'CQL'  # the offline algorithm

    refer_with_optimal_pi: bool = True  # use the best foregoing policy as the reference policy
    update_n: int = 1  # the interval for updating the reference policy

@pyrallis.wrap()
def train(config: TrainConfig):
    config = load_train_config_auto(config, 'finetune', 'O2SAC')

    grad_clip = False
    if not config.refer_with_optimal_pi:
        config.end_threshold /= 4
        if 'expert' in config.env and 'hopper' in config.env:
            grad_clip = True
            config.update_n = 10

    env = gym.make(config.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    dataset = d4rl.qlearning_dataset(env)

    if config.normalize_reward:
        modify_reward(dataset, config.env, 10.0, -5.0)

    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)

    max_action = float(env.action_space.high[0])

    # Set seeds
    seed = config.seed
    set_seed(seed, env)

    critic_1 = CriticFunction(state_dim, action_dim, layer_norm=config.critic_layernorm,
                              n_hidden=config.q_n_hidden_layers).to(config.device)
    critic_1_optimizer = torch.optim.Adam(list(critic_1.parameters()), config.qf_lr)

    critic_2 = CriticFunction(state_dim, action_dim, layer_norm=config.critic_layernorm,
                              n_hidden=config.q_n_hidden_layers).to(config.device)
    critic_2_optimizer = torch.optim.Adam(list(critic_2.parameters()), config.qf_lr)

    actor = TanhGaussianPolicy(
        state_dim,
        action_dim,
        max_action,
        log_std_multiplier=config.policy_log_std_multiplier,
        orthogonal_init=config.orthogonal_init
    ).to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), config.policy_lr)
    target_entropy = -np.prod(env.action_space.shape).item()
    log_alpha = np.log(config.alpha)
    log_lmbda = np.log(config.lmbda)

    trainer = SAC(critic_1, critic_1_optimizer, critic_2, critic_2_optimizer, actor, actor_optimizer, target_entropy,
                  max_action, config.discount, config.alpha_multiplier, config.alpha_lr, config.target_update_period,
                  config.soft_target_update_rate, config.use_automatic_entropy_tuning, log_alpha, config.threshold,
                  config.end_threshold, config.end_steps, config.double_critic, log_lmbda, config.loss_tau,
                  config.device)

    config.load_model = f'../offline/model/{config.offline_alg}/{config.env}/{config.seed}/model.pth'
    if os.path.exists(config.load_model):
        policy_file = Path(config.load_model)
        state_dict = torch.load(policy_file)
        trainer.load_offline(state_dict)
    else:
        raise FileExistsError

    print(f"--------After offline training--------")
    eval_scores = eval_actor(
        env,
        trainer.actor,
        device=config.device,
        n_episodes=config.n_episodes,
        seed=config.seed,
    )
    eval_score = eval_scores.mean()
    normalized_eval_score = env.get_normalized_score(eval_score) * 100.0
    # explore_scores = eval_actor_explore(env, trainer.actor, config.device, config.n_episodes, config.seed)
    # explore_score = explore_scores.mean()
    # normalized_explore_score = env.get_normalized_score(explore_score) * 100.0
    print(
        f"Evaluation over {config.n_episodes} episodes: "
        f"eval: {eval_score:.3f} , D4RL score: {normalized_eval_score:.3f} "
        # f"explore: {explore_score:.3f} , D4RL score: {normalized_explore_score:.3f}"
    )
    print("---------------------------------------")

    offline_eval_score = normalized_eval_score

    writer = SummaryWriter(os.path.join(str(config.save_dir), f'{config.offline_alg}_toSAC', str(config.env),
                                        str(config.seed), str(config.refer_with_optimal_pi)), write_to_disk=True)

    writer.add_scalar('reconstruct_return', normalized_eval_score, 0)

    save_model = os.path.join('../finetune/model', f'{config.offline_alg}_toSAC', config.env, str(config.seed))

    replay_buffer = ReplayBuffer(state_dim, action_dim, config.buffer_size, config.device)
    replay_buffer.load_d4rl_dataset(dataset)

    # Off-policy Evaluation
    ope_path = save_model + '/ope_model'
    if os.path.exists(ope_path + '/model.pth'):
        policy_file = Path(ope_path + '/model.pth')
        trainer.load_state_dict(torch.load(policy_file))
    else:
        os.makedirs(ope_path, exist_ok=True)
        for i in tqdm.tqdm(range(config.ope_steps)):
            batch = replay_buffer.sample(config.batch_size)
            ope_log_dict = trainer.ope(batch)
            if (i + 1) % config.log_freq == 0:
                for k, v in ope_log_dict.items():
                    writer.add_scalar(f'{k}', v, i)
        torch.save(trainer.state_dict(), ope_path + '/model.pth')

    # Value Alignment
    reconstruct_path = save_model + '/reconstruct_model'
    if os.path.exists(reconstruct_path + '/model.pth'):
        policy_file = Path(reconstruct_path + '/model.pth')
        trainer.load_state_dict(torch.load(policy_file))
    else:
        os.makedirs(reconstruct_path, exist_ok=True)
        for i in tqdm.tqdm(range(config.align_steps)):
            batch = replay_buffer.sample(config.batch_size)
            refine_log_dict = trainer.align_value(batch)
            if (i + 1) % config.log_freq == 0:
                for k, v in refine_log_dict.items():
                    writer.add_scalar(f'{k}', v, i)
                evaluate_reward = eval_actor(env, trainer.actor, config.device, config.n_episodes, config.seed)
                normalized_eval_score = env.get_normalized_score(evaluate_reward.mean()) * 100.0
                writer.add_scalar('reconstruct_return', normalized_eval_score, i)

        trainer.target_critic_1 = copy.deepcopy(trainer.critic_1)
        trainer.target_critic_2 = copy.deepcopy(trainer.critic_2)

        torch.save(trainer.state_dict(), reconstruct_path + '/model.pth')

    del replay_buffer
    # online train
    print("-----------After reconstruct-----------")
    eval_scores = eval_actor(
        env,
        trainer.actor,
        device=config.device,
        n_episodes=config.n_episodes,
        seed=config.seed,
    )
    eval_score = eval_scores.mean()
    normalized_eval_score = env.get_normalized_score(eval_score) * 100.0
    # explore_scores = eval_actor_explore(env, trainer.actor, config.device, config.n_episodes, config.seed)
    # explore_score = explore_scores.mean()
    # normalized_explore_score = env.get_normalized_score(explore_score) * 100.0
    print(
        f"Evaluation over {config.n_episodes} episodes: "
        f"eval: {eval_score:.3f} , D4RL score: {normalized_eval_score:.3f} "
        # f"explore: {explore_score:.3f} , D4RL score: {normalized_explore_score:.3f}"
    )
    print("---------------------------------------")
    writer.add_scalar('evaluate_reward', eval_score, 0)
    writer.add_scalar('normalized_return', normalized_eval_score, 0)
    # writer.add_scalar('explore_reward', explore_score, 0)
    # writer.add_scalar('normalized_explore_return', normalized_explore_score, 0)

    if normalized_eval_score > offline_eval_score:
        trainer.offline_actor = copy.deepcopy(trainer.actor)
        last_eval_score = normalized_eval_score
    else:
        last_eval_score = offline_eval_score

    data_dict = {}
    eval_return_list = []
    eval_return_list.append(offline_eval_score)

    online_replay_buffer = Online_ReplayBuffer(state_dim, action_dim, config.buffer_size, config.device)
    online_replay_buffer.initial(config.buffer_load, dataset)

    evaluate_num = 0
    total_steps = 0
    episode_num = 0
    goal_achieved = False
    while total_steps < config.online_steps:
        s = env.reset()
        episode_steps, episode_rewards = 0, 0
        done = False
        while not done:
            episode_steps += 1
            a = trainer.choose_action(s)
            action = a
            s_, r, done, info = env.step(action)

            if not goal_achieved:
                goal_achieved = is_goal_reached(r, info)

            episode_rewards += r

            dw = False
            if done:
                episode_num += 1
                writer.add_scalar('episode_rewards', episode_rewards, episode_num)
                if "TimeLimit.truncated" not in info:
                    dw = True

            if config.normalize_reward:
                r = modify_reward_online(config.env, r, config.reward_scale, config.reward_bias)

            online_replay_buffer.add_transition(s, a, r, s_, dw)

            s = s_
            total_steps += 1

            if total_steps > config.warmup_steps and online_replay_buffer.size() > config.batch_size:
                for times in range(10):
                    batch = online_replay_buffer.sample(config.batch_size)
                    update_info = trainer.update(batch, total_steps, grad_clip, times)

                for k, v in update_info.items():
                    writer.add_scalar(f'{k}', v, total_steps)

            if total_steps % config.eval_freq == 0:
                evaluate_num += 1
                evaluate_reward = eval_actor(env, trainer.actor, config.device, config.n_episodes,
                                             config.seed).mean()
                normalized_eval_score = env.get_normalized_score(evaluate_reward) * 100.0
                explore_reward = eval_actor_explore(env, trainer.actor, config.device, config.n_episodes,
                                                    config.seed).mean()
                normalized_explore_score = env.get_normalized_score(explore_reward) * 100.0
                writer.add_scalar('evaluate_reward', evaluate_reward, total_steps)
                writer.add_scalar('normalized_return', normalized_eval_score, total_steps)
                # writer.add_scalar('explore_reward', explore_reward, total_steps)
                # writer.add_scalar('normalized_explore_return', normalized_explore_score, total_steps)

                if config.refer_with_optimal_pi:
                    if normalized_eval_score > last_eval_score:
                        last_eval_score = normalized_eval_score
                        trainer.offline_actor = copy.deepcopy(trainer.actor)
                else:
                    if total_steps % (config.update_n * config.eval_freq) == 0:
                        trainer.offline_actor = copy.deepcopy(trainer.actor)

                eval_return_list.append(normalized_eval_score)

    torch.save(trainer.state_dict(), save_model + f'/eventual_model_{config.refer_with_optimal_pi}.pth')
    file_path = save_model + f'/data_{config.refer_with_optimal_pi}.json'

    steps_list = [i * config.eval_freq for i in list(range(0, len(eval_return_list)))]
    data_dict['steps'] = steps_list
    data_dict['eval_returns'] = eval_return_list
    with open(file_path, "w") as json_file:
        json.dump(data_dict, json_file, indent=2)


if __name__ == "__main__":
    train()
