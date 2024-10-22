# source: https://github.com/Lei-Kun/Uni-O4/blob/master/ppo_finetune/main.py
import os
import yaml
import copy
import json
import argparse

import gym
import torch
import numpy as np
import d4rl.gym_mujoco

from tqdm import tqdm
from collections import deque
from tensorboardX import SummaryWriter

from online_agent.ppo.ppo import PPO
from online_agent.ppo.normalization import Normalization
from online_agent.ppo.online_buffer import ReplayBuffer
from online_agent.ppo.ppo_utils import evaluate_policy
from online_agent.ppo.ppo_net import ValueLearner
from online_agent.ppo.offline_buffer import OfflineReplayBuffer

def str2bool(input_value):
    if isinstance(input_value, bool):
        return input_value
    if input_value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif input_value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO")
    # base hyperparameters and settings for online ppo
    parser.add_argument("--env_name", type=str, default='pen-binary-v0', help="training env")
    parser.add_argument("--seed", type=int, default=0, help="run with a fixed seed")
    parser.add_argument("--max_train_steps", type=int, default=int(2.5e5), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=int, default=2500,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=128, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=256,
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--v_hidden_width", type=int, default=256,
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--gpu", default=0, type=int, help='id of gpu')
    parser.add_argument("--depth", type=int, default=3, help="The number of layer in MLP")
    parser.add_argument("--v_depth", type=int, default=3, help="The number of layer in MLP")
    parser.add_argument("--lr_a", type=float, default=3e-5, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=2e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")

    # tricks usually used in online ppo
    parser.add_argument("--use_adv_norm", type=str2bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=str2bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=str2bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=str2bool, default=False,
                        help="Trick 4:reward scaling")  # if use here, please retrain the value function
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=str2bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=str2bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=str2bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=str2bool, default=False, help="Trick 10: tanh activation function")
    # tricks usually used in online ppo

    # setup for fine-tuning
    parser.add_argument("--offline_alg", type=str, default="IQL", help="Load the offline policy")
    parser.add_argument("--is_shuffle", type=str2bool, default=True, help="shuffle the dataset")
    parser.add_argument("--r_scale", default=1., type=float, help='the weight of Q loss')
    parser.add_argument("--is_clip_value", default=False, type=str2bool,
                        help='Asynchronous Update: train critic then update policy')
    parser.add_argument("--scale_strategy", default='normal', type=str,
                        help='reward scaling technique: dynamic/normal/number(0.1)')
    parser.add_argument("--is_decay_pi", default=False, type=str2bool, help='decay the update of target policy')
    parser.add_argument("--tau", default=5e-3, type=float)
    parser.add_argument("--std_upper_bound", default=0, type=float)

    parser.add_argument("--alpha", default=2.0, type=float, help='the factor of auxiliary advantage')
    parser.add_argument("--end_steps", default=250000, type=int, help='the steps of decay steps')
    parser.add_argument("--use_auxi", type=str2bool, default=True, help='use the auxiliary advantage')
    parser.add_argument("--refer_with_optimal_pi", type=str2bool, default=True, help='use the best foregoing policy as the reference policy')
    parser.add_argument("--update_n", default=1, type=int, help='the interval for updating the reference policy')
    parser.add_argument("--eval_episodes", default=10, type=int, help='How many episodes run during evaluation')

    args = parser.parse_args()

    env_higher = "_".join(args.env_name.split("-")[:1]).lower().replace("-", "_")
    env_lower = "_".join(args.env_name.split("-")[1:]).lower().replace("-", "_")

    with open(f'../config/finetune/O2PPO/{env_higher}/{env_lower}.yaml', 'r') as file:
        loaded_args = yaml.safe_load(file)

    for key, value in loaded_args.items():
        setattr(args, key, value)

    if not args.refer_with_optimal_pi:
        if 'expert' in args.env_name and 'hopper' in args.env_name:
            args.update_n = 4

    env_name = args.env_name

    seed = args.seed

    env = gym.make(env_name)
    env_evaluate = gym.make(env_name)
    # Set random seed
    env.seed(seed)
    env.action_space.seed(seed)
    env_evaluate.seed(seed)
    env_evaluate.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_action = float(env.action_space.high[0])
    args.max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
    model_path = os.path.join(f'./model/{args.offline_alg}_toPPO', args.env_name, str(args.seed))

    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)

    # summarywriter logger
    comment = args.env_name + '_' + str(args.seed)
    writer = SummaryWriter(log_dir=f'./log/{args.offline_alg}_toPPO/{args.env_name}/{args.seed}',
                           write_to_disk=True)

    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training

    replay_buffer = ReplayBuffer(args, device=device)
    agent = PPO(args, device)

    dataset = env.get_dataset()

    offline_buffer = OfflineReplayBuffer(device, args.state_dim, args.action_dim, len(dataset['actions']) - 1)
    offline_buffer.load_dataset(dataset=dataset, clip=True)
    if args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = offline_buffer.reward_normalize(args.gamma, args.scale_strategy)
    elif args.scale_strategy == 'number':
        offline_buffer.reward_normalize(args.gamma, args.scale_strategy)

    offline_buffer.compute_return(args.gamma)

    if args.is_shuffle:
        offline_buffer.shuffle()
    if args.use_state_norm:
        mean, std = offline_buffer.normalize_state()
    else:
        mean, std = 0., 1.

    state_norm = Normalization(shape=args.state_dim, mean=mean, std=std)  # Trick 2:state normalization

    v_path = os.path.join(model_path, 'value_{}.pt'.format(args.scale_strategy))
    if not os.path.exists(v_path):
        value = ValueLearner(args, value_lr=1e-4, batch_size=512)
        for step in tqdm(range(int(1e6)), desc='value upladating ......'):
            value_loss = value.update(offline_buffer)
            if step % int(2e4) == 0:
                print(f"Step: {step}, Loss: {value_loss:.4f}")
                writer.add_scalar('value_loss', value_loss, global_step=(step + 1))
        value.save(v_path)
    agent.load_value(v_path)
    agent.load_actor(f'../offline/model/{args.offline_alg}/{args.env_name}/{args.seed}/model.pth')

    if 'hopper' in args.env_name and 'expert' in args.env_name:
        agent.actor.reset_std(np.log(0.05))
        agent.reference_actor.reset_std(np.log(0.05))

    evaluate_score = evaluate_policy(args, env_evaluate, agent, state_norm, True,
                                     eval_episodes=args.eval_episodes)
    print(f'------After offline training, d4rl score: {evaluate_score} ------')
    writer.add_scalar('normalized_return', evaluate_score, global_step=0)

    reference_score = evaluate_score

    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    total_episode_r = deque(maxlen=10)
    episode_reward = 0
    scores = []
    d4rl_scores = []
    d4rl_scores.append(evaluate_score)
    actor_losses, critic_losses, values, first_values = [], [], [], []
    while total_steps < args.max_train_steps:
        s = env.reset()
        if args.use_state_norm:
            s = state_norm(s, update=False)
        if args.use_reward_scaling:
            reward_scaling.reset()
        episode_steps = 0
        done = False

        total_episode_r.append(episode_reward)
        episode_reward = 0

        while not done:
            episode_steps += 1
            action, a_logprob = agent.choose_action(s)  # Action and the corresponding log probability
            s_, r, done, info = env.step(action)

            if args.use_state_norm:
                s_ = state_norm(s_, update=False)
            if args.use_reward_norm:
                r = reward_norm(r)
            elif args.use_reward_scaling:
                r = reward_scaling(r)[0]
            episode_reward += r

            if done and episode_steps != args.max_episode_steps:
                dw = True
            else:
                dw = False

            replay_buffer.store(s, action, a_logprob, r * args.r_scale, s_, dw, done)
            s = s_
            total_steps += 1

            if replay_buffer.count == args.batch_size:
                actor_loss, critic_loss = agent.update(replay_buffer, total_steps)
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)

                replay_buffer.count = 0

                writer.add_scalar('actor_loss', actor_loss, global_step=total_steps)
                writer.add_scalar('critic_loss', critic_loss, global_step=total_steps)

            if total_steps % args.evaluate_freq == 0:
                evaluate_num += 1
                avg_return, d4rl_score, value_score, first_score = agent.off_evaluate(args.env_name, args.seed,
                                                                                      state_norm.running_ms.mean,
                                                                                      state_norm.running_ms.std,
                                                                                      eval_episodes=args.eval_episodes)

                if args.refer_with_optimal_pi:
                    if d4rl_score > reference_score:
                        reference_score = d4rl_score
                        agent.reference_actor = copy.deepcopy(agent.actor)
                else:
                    if total_steps % (args.update_n * args.evaluate_freq) == 0:
                        agent.reference_actor = copy.deepcopy(agent.actor)

                scores.append(avg_return)
                d4rl_scores.append(d4rl_score)
                values.append(value_score)
                first_values.append(first_score)


                writer.add_scalar('evaluate_reward', avg_return, global_step=total_steps)
                writer.add_scalar('normalized_return', d4rl_score, global_step=total_steps)

    agent.save(model_path)

    data_dict = {}
    data_dict['score'] = d4rl_scores
    file_path = model_path + f'/data_{args.refer_with_optimal_pi}.json'
    with open(file_path, "w") as json_file:
        json.dump(data_dict, json_file, indent=2)