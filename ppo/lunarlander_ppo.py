import os
import yaml
import torch
import argparse
import numpy as np

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
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


def _train(env, configs):
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=500e3)
    model.save(configs['model'])


def _load_model(configs):
    model = PPO.load(configs['model'])

    return model


def _make_env(render='rgb_array'):
    env = gym.make("LunarLander-v2", continuous=False, render_mode=render)
    env.reset(seed=47)

    return env


if __name__ == "__main__":
    args = _get_arguments()
    configs = _load_configs(args.config)

    print(f"Config loaded: {configs}")

    np.random.seed(47)
    torch.manual_seed(47)
    torch.cuda.manual_seed_all(47)

    # Make PyTorch deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # env = _make_env()
    env_vec = DummyVecEnv([_make_env for _ in range(1)])
    
    do_train = not os.path.exists(configs['model'] + ".zip") or configs['overwrite']

    if do_train:
        _train(env_vec, configs)
    
    model = _load_model(configs)

    env = _make_env("human")
    obs, _ = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, truncated, info = env.step(action)
        env.render()

        if dones or truncated:
            obs, _ = env.reset()
