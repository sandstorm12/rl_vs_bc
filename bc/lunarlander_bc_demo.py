import yaml
import torch
import argparse
import numpy as np
import gymnasium as gym

from lunarlander_bc import MLP_BC


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
    env = gym.make("LunarLander-v2", continuous=False, render_mode='human')

    return env


def _load_model(model_path):
    model = MLP_BC(8, 4)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model


def _demo(configs):
    model = _load_model(configs['model'])
    env = _load_env()

    mean = np.array(configs['mean'])
    std = np.array(configs['std'])

    obs, _ = env.reset()
    while True:
        obs = (obs - mean) / std
        obs = torch.tensor(obs, dtype=torch.float32)

        action = model(obs)

        action = torch.argmax(action).item()
        
        obs, rewards, dones, truncated, info = env.step(action)
        env.render()

        if dones or truncated:
            obs, _ = env.reset()


if __name__ == '__main__':
    args = _get_arguments()
    configs = _load_configs(args.config)

    print(f"Config loaded: {configs}")

    _demo(configs)
