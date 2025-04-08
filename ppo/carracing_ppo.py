import os
import yaml
import argparse

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


def _train(env_vec, configs):
    model = PPO("CnnPolicy", env_vec, verbose=1)
    model.learn(total_timesteps=1000e3)
    model.save(configs['model'])


def _load_model(configs):
    model = PPO.load(configs['model'])

    return model


def _make_env(render='rgb_array'):
    return gym.make("CarRacing-v2", continuous=False, render_mode=render)


if __name__ == "__main__":
    args = _get_arguments()
    configs = _load_configs(args.config)

    print(f"Config loaded: {configs}")

    env_vec = DummyVecEnv([_make_env for _ in range(4)])
    
    do_train = not os.path.exists(configs['model'] + ".zip") or configs['overwrite']

    if do_train:
        _train(env_vec, configs)
    
    model = _load_model(configs)

    env = _make_env("human")
    obs, _ = env.reset()
    steps = 0
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, truncated, info = env.step(action)
        env.render()
        steps += 1

        if steps > 1000 or dones and truncated:
            obs, _ = env.reset()
            steps = 0

