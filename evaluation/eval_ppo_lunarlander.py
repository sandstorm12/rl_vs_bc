import os
import yaml
import torch
import argparse
import numpy as np

from tqdm import tqdm

from stable_baselines3 import PPO
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


def _load_model(configs):
    model = PPO.load(configs['model'])

    return model


if __name__ == "__main__":
    args = _get_arguments()
    configs = _load_configs(args.config)

    print(f"Config loaded: {configs}")

    env_vec = make_vec_env("LunarLander-v2", n_envs=1, seed=42)
    
    model = _load_model(configs)
    torch.manual_seed(42)

    episodes = 0
    rewards_all = []

    bar = tqdm(range(configs['num_episodes']))

    obs = env_vec.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env_vec.step(action)
        # env_vec.render("human")

        rewards_all.append(rewards)

        if dones:
            obs = env_vec.reset()
            episodes += 1
            bar.update(1)
            if episodes == configs['num_episodes']:
                break
 
    print("Avg reward", np.sum(rewards_all) / configs['num_episodes'])
