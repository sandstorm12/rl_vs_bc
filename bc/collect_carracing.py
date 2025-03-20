import os
import cv2
import yaml
import shelve
import pickle
import argparse

import gymnasium as gym

from tqdm import tqdm
from stable_baselines3 import PPO


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


def _load_env():
    env = gym.make("CarRacing-v2", continuous=False, render_mode='rgb_array')

    return env


def _store_artifact(demonstrations, configs):
    with open(configs['store_path'], 'wb') as f:
        pickle.dump(demonstrations, f)


def _collect(configs):
    if os.path.exists(configs['store_path']) and not configs['overwrite']:
        return
    
    model = _load_model(configs)
    env = _load_env()

    with shelve.open(configs['store_path']) as storage:
        obs, _ = env.reset()
        episode_length = 0
        for idx_sample in tqdm(range(configs['num_samples'])):
            action, _states = model.predict(obs)
            obs_new, rewards, dones, truncated, info = env.step(action)

            if configs['visualize']:
                cv2.imshow("Action", cv2.resize(obs, (512, 512)))
                if cv2.waitKey(1) == ord('q'):
                    break

            storage[str(idx_sample)] = (
                obs, obs_new, action,
                rewards, dones, truncated, info)

            if dones or episode_length == configs['max_episode_length']:
                obs_new, _ = env.reset()
                episode_length = 0

            obs = obs_new
            episode_length += 1


if __name__ == "__main__":
    args = _get_arguments()
    configs = _load_configs(args.config)

    print(f"Config loaded: {configs}")

    _collect(configs)
