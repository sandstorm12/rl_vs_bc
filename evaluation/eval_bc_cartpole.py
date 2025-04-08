import sys
sys.path.append("..")

import yaml
import torch
import argparse
import numpy as np
import gymnasium as gym

from tqdm import tqdm
from bc.cartpole_bc import MLP_BC
from stable_baselines3.common.env_util import make_vec_env


def _get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c', '--config',
        help='Path to the config file',
        type=str,
        required=True,
    )

    args = parser.parse_args()

    return args


def _load_configs(path):
    with open(path, 'r') as yaml_file:
        configs = yaml.safe_load(yaml_file)

    return configs


def _load_env():
    env = make_vec_env("CartPole-v1", n_envs=1, seed=42)

    return env


def _load_model(model_path):
    model = MLP_BC(4, 2)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model


def _demo(configs):
    model = _load_model(configs['model'])
    torch.manual_seed(42)

    env = _load_env()

    mean = np.array(configs['mean'])
    std = np.array(configs['std'])

    episodes = 0
    rewards_all = []

    bar = tqdm(range(configs['num_episodes']))

    obs = env.reset()
    while True:
        obs = (obs - mean) / std
        obs = torch.tensor(obs, dtype=torch.float32)

        action = model(obs)

        action = torch.argmax(action).item()
        
        obs, rewards, dones, info = env.step([action])
        # env.render('human')

        rewards_all.append(rewards)

        if dones:
            obs = env.reset()
            episodes += 1
            bar.update(1)
            if episodes == configs['num_episodes']:
                break

    print("Avg reward", np.sum(rewards_all) / configs['num_episodes'])


if __name__ == '__main__':
    args = _get_arguments()
    configs = _load_configs(args.config)

    print(f"Config loaded: {configs}")

    _demo(configs)
