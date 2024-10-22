# source: https://github.com/tinkoff-ai/CORL/blob/main/algorithms/offline/cql.py
import os
import sys
import json
import tqdm
import pyrallis

sys.path.append('..')

import torch
import numpy as np

import d4rl
import gym

from pathlib import Path
from dataclasses import dataclass
from tensorboardX import SummaryWriter

from net import TanhGaussianPolicy, GaussianPolicy, CriticFunction
from dataset_utils import ReplayBuffer
from utils import modify_reward, compute_mean_std, normalize_states, wrap_env, set_seed, eval_actor, \
    load_train_config_auto
from offline_agent.cql import ContinuousCQL


@dataclass
class TrainConfig:
    # Experiment
    device: str = "cuda"
    env: str = "halfcheetah-medium-expert-v2"  # OpenAI gym environment name
    seed: int = 1  # Sets Gym, PyTorch and Numpy seeds
    eval_freq: int = int(5e3)  # How often (time steps) we evaluate
    n_episodes: int = 10  # How many episodes run during evaluation
    max_timesteps: int = int(1e6)  # Max time steps to run environment
    load_model: str = ""  # Model load file name, "" doesn't load

    # CQL
    buffer_size: int = 2_000_000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    alpha_multiplier: float = 1.0  # Multiplier for alpha in loss
    use_automatic_entropy_tuning: bool = True  # Tune entropy
    backup_entropy: bool = False  # Use backup entropy
    policy_lr: float = 3e-5  # Policy learning rate
    qf_lr: float = 3e-4  # Critics learning rate
    soft_target_update_rate: float = 5e-3  # Target network update rate
    target_update_period: int = 1  # Frequency of target nets updates
    cql_n_actions: int = 10  # Number of sampled actions
    cql_importance_sample: bool = True  # Use importance sampling
    cql_lagrange: bool = False  # Use Lagrange version of CQL
    cql_target_action_gap: float = -1.0  # Action gap
    cql_temp: float = 1.0  # CQL temperature
    cql_alpha: float = 10.0  # Minimal Q weight
    cql_max_target_backup: bool = False  # Use max target backup
    cql_clip_diff_min: float = -np.inf  # Q-function lower loss clipping
    cql_clip_diff_max: float = np.inf  # Q-function upper loss clipping
    orthogonal_init: bool = True  # Orthogonal initialization
    normalize: bool = True  # Normalize states
    normalize_reward: bool = False  # Normalize reward
    q_n_hidden_layers: int = 3  # Number of hidden layers in Q networks
    reward_scale: float = 1.0  # Reward scale for normalization
    reward_bias: float = 0.0  # Reward bias for normalization

    # AntMaze hacks
    bc_steps: int = int(0)  # Number of BC steps at start
    policy_log_std_multiplier: float = 1.0

    actor_model: str = 'tanh'  # use squashed Gaussian distribution

    save_dir = '../offline/log'  # the directory saving the training log
    json_load = True  # save the results with a .json file
    alg = 'CQL'  # offline algorithm


@pyrallis.wrap()
def train(config: TrainConfig):
    if 'antmaze' in config.env:
        config = load_train_config_auto(config, 'offline', 'cql')

    env = gym.make(config.env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    dataset = d4rl.qlearning_dataset(env)

    if config.normalize_reward:
        modify_reward(
            dataset,
            config.env,
            reward_scale=config.reward_scale,
            reward_bias=config.reward_bias,
        )

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
    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
    )
    replay_buffer.load_d4rl_dataset(dataset)

    max_action = float(env.action_space.high[0])

    # Set seeds
    seed = config.seed
    set_seed(seed, env)

    critic_1 = CriticFunction(state_dim, action_dim, n_hidden=config.q_n_hidden_layers).to(config.device)
    critic_2 = CriticFunction(state_dim, action_dim, n_hidden=config.q_n_hidden_layers).to(config.device)
    critic_1_optimizer = torch.optim.Adam(list(critic_1.parameters()), config.qf_lr)
    critic_2_optimizer = torch.optim.Adam(list(critic_2.parameters()), config.qf_lr)

    if config.actor_model == 'tanh':
        actor = TanhGaussianPolicy(
            state_dim,
            action_dim,
            max_action,
            log_std_multiplier=config.policy_log_std_multiplier,
            orthogonal_init=config.orthogonal_init
        ).to(config.device)
    else:
        actor = GaussianPolicy(
            state_dim,
            action_dim,
            max_action,
            n_hidden=3
        ).to(config.device)

    actor_optimizer = torch.optim.Adam(actor.parameters(), config.policy_lr)

    kwargs = {
        "critic_1": critic_1,
        "critic_2": critic_2,
        "critic_1_optimizer": critic_1_optimizer,
        "critic_2_optimizer": critic_2_optimizer,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "discount": config.discount,
        "soft_target_update_rate": config.soft_target_update_rate,
        "device": config.device,
        # CQL
        "target_entropy": -np.prod(env.action_space.shape).item(),
        "alpha_multiplier": config.alpha_multiplier,
        "use_automatic_entropy_tuning": config.use_automatic_entropy_tuning,
        "backup_entropy": config.backup_entropy,
        "policy_lr": config.policy_lr,
        "qf_lr": config.qf_lr,
        "bc_steps": config.bc_steps,
        "target_update_period": config.target_update_period,
        "cql_n_actions": config.cql_n_actions,
        "cql_importance_sample": config.cql_importance_sample,
        "cql_lagrange": config.cql_lagrange,
        "cql_target_action_gap": config.cql_target_action_gap,
        "cql_temp": config.cql_temp,
        "cql_alpha": config.cql_alpha,
        "cql_max_target_backup": config.cql_max_target_backup,
        "cql_clip_diff_min": config.cql_clip_diff_min,
        "cql_clip_diff_max": config.cql_clip_diff_max,
    }

    print("---------------------------------------")
    print(f"Training CQL, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize actor
    trainer = ContinuousCQL(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    writer = SummaryWriter(os.path.join(config.save_dir, config.alg, config.env, str(config.seed)), write_to_disk=True)

    if config.json_load:
        data_dict = {}
        eval_return_list = []
        # expl_return_list = []

    model_dir = os.path.join('../offline/model', config.alg, config.env, str(config.seed))
    os.makedirs(model_dir, exist_ok=True)

    for t in tqdm.tqdm(range(int(config.max_timesteps))):
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        log_dict = trainer.train(batch)
        for k, v in log_dict.items():
            writer.add_scalar(f'{k}', v, t)
        # Evaluate episode
        if (t + 1) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            eval_scores = eval_actor(env, actor, config.device, config.n_episodes, config.seed)
            eval_score = eval_scores.mean()
            normalized_eval_score = env.get_normalized_score(eval_score) * 100.0
            # explore_scores = eval_actor_explore(env, actor, config.device, config.n_episodes, config.seed)
            # explore_score = explore_scores.mean()
            # normalized_explore_score = env.get_normalized_score(explore_score) * 100.0
            print("---------------------------------------")
            print(
                f"Evaluation over {config.n_episodes} episodes: "
                f"eval: {eval_score:.3f} , D4RL score: {normalized_eval_score:.3f} "
                # f"explore: {explore_score:.3f} , D4RL score: {normalized_explore_score:.3f} "
            )
            print("---------------------------------------")

            writer.add_scalar('eval_normalized_return', normalized_eval_score, t)
            # writer.add_scalar('explore_normalized_return', normalized_explore_score, t)

            if config.json_load:
                eval_return_list.append(normalized_eval_score)
                # expl_return_list.append(normalized_explore_score)

    torch.save(trainer.state_dict(), model_dir + '/model.pth')

    if config.json_load:
        steps_list = [i * config.eval_freq for i in list(range(1, len(eval_return_list) + 1))]
        data_dict['steps'] = steps_list
        data_dict['eval_returns'] = eval_return_list
        # data_dict['expl_returns'] = expl_return_list
        file_path = model_dir + '/data.json'
        with open(file_path, "w") as json_file:
            json.dump(data_dict, json_file, indent=2)

        print("data has been loaded in:", file_path)


if __name__ == "__main__":
    train()
