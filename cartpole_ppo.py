import os
import yaml
import argparse

import gymnasium as gym

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


def _train(env_vec, configs):
    model = PPO("MlpPolicy", env_vec, verbose=1)
    model.learn(total_timesteps=25e3)
    model.save(configs['model'])


def _load_model(configs):
    model = PPO.load(configs['model'])

    return model


if __name__ == "__main__":
    args = _get_arguments()
    configs = _load_configs(args.config)

    print(f"Config loaded: {configs}")

    env_vec = make_vec_env("CartPole-v1", n_envs=1)
    
    do_train = not os.path.exists(configs['model'] + ".zip") or configs['overwrite']

    if do_train:
        _train(env_vec, configs)
    
    model = _load_model(configs)

    obs = env_vec.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env_vec.step(action)
        env_vec.render("human")
