# source: https://github.com/tinkoff-ai/CORL/blob/main/algorithms/offline/td3_bc.py
import os
import sys
import json
import tqdm
import pyrallis

sys.path.append('..')

import torch

import d4rl
import gym

from pathlib import Path
from dataclasses import dataclass
from tensorboardX import SummaryWriter

from net import DeterministicPolicy, CriticFunction
from dataset_utils import ReplayBuffer
from utils import modify_reward, compute_mean_std, normalize_states, wrap_env, set_seed, eval_actor, \
    load_train_config_auto
from offline_agent.td3_bc import TD3_BC


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
    # TD3
    buffer_size: int = 2_000_000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount ffor
    qf_lr: float = 3e-4  # Critic learning rate
    actor_lr: float = 3e-4  # Actor learning rate
    explore_noise: float = 0.1  # Std of Gaussian exploration noise
    tau: float = 0.005  # Target network update rate
    policy_noise: float = 0.2  # Noise added to target actor during critic update
    noise_clip: float = 0.5  # Range to clip target actor noise
    policy_freq: int = 2  # Frequency of delayed actor updates
    # TD3 + BC
    alpha: float = 2.5  # Coefficient for Q function in actor loss
    normalize: bool = True  # Normalize states
    normalize_reward: bool = False  # Normalize reward

    save_dir = '../offline/log'  # the directory saving the training log
    json_load = True  # save the results with a .json file
    alg = 'TD3_BC'  # offline algorithm


@pyrallis.wrap()
def train(config: TrainConfig):
    if 'antmaze' in config.env:
        config = load_train_config_auto(config, 'offline', 'td3_bc')

    env = gym.make(config.env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    dataset = d4rl.qlearning_dataset(env)

    if config.normalize_reward:
        modify_reward(dataset, config.env)

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

    actor = DeterministicPolicy(state_dim, action_dim, max_action).to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)

    critic_1 = CriticFunction(state_dim, action_dim).to(config.device)
    critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=config.qf_lr)
    critic_2 = CriticFunction(state_dim, action_dim).to(config.device)
    critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=config.qf_lr)

    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "critic_1": critic_1,
        "critic_1_optimizer": critic_1_optimizer,
        "critic_2": critic_2,
        "critic_2_optimizer": critic_2_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": config.device,
        # TD3
        "policy_noise": config.policy_noise * max_action,
        "noise_clip": config.noise_clip * max_action,
        "policy_freq": config.policy_freq,
        # TD3 + BC
        "alpha": config.alpha,
    }

    print("---------------------------------------")
    print(f"Training TD3 + BC, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize actor
    trainer = TD3_BC(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    writer = SummaryWriter(os.path.join(config.save_dir, config.alg, config.env, str(config.seed)),
                           write_to_disk=True)

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
            # explore_scores = eval_deterministic_actor_explore(env, actor, config.device, config.n_episodes, config.seed,
            #                                                   config.explore_noise, action_dim)
            # explore_score = explore_scores.mean()
            # normalized_explore_score = env.get_normalized_score(explore_score) * 100.0
            print("---------------------------------------")
            print(
                f"Evaluation over {config.n_episodes} episodes: "
                f"eval: {eval_score:.3f} , D4RL score: {normalized_eval_score:.3f} "
                # f"explore: {explore_score:.3f} , D4RL score: {normalized_explore_score:.3f}"
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
