import os
import yaml
import torch
import pickle
import argparse
import numpy as np
import  torch.nn.functional as F
import gymnasium as gym

from torch.utils.data import Dataset, DataLoader



class MLP_BC(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(MLP_BC, self).__init__()
        self.layer1 = torch.nn.Linear(input_dim, hidden_dim)
        self.layer2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.layer22 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = torch.nn.Linear(hidden_dim, output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer22(x))
        x = self.layer3(x)
        
        return x
    

class BCDataset(Dataset):
    def __init__(self, observations, actions):
        self._observations = observations
        self._actions = actions

    def __len__(self):
        return len(self._observations)
    
    def __getitem__(self, idx):
        return self._observations[idx], self._actions[idx]


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

    obs, _ = env.reset()
    while True:
        obs = (obs - MEAN) / STD
        obs = torch.tensor(obs, dtype=torch.float32)

        action = model(obs)

        action = torch.argmax(action).item()
        
        obs, rewards, dones, truncated, info = env.step(action)
        env.render()

        if dones or truncated:
            obs, _ = env.reset()


MEAN = np.array([ 0.09450758, 0.59836125, 0.02613196, -0.12162905, 0.01558024, 0.0031787, 0.1349, 0.1238])
STD = np.array([0.19597459, 0.46255904, 0.13085105, 0.08376966, 0.10467913, 0.12128206, 0.34161037, 0.32934433])

if __name__ == '__main__':
    args = _get_arguments()
    configs = _load_configs(args.config)

    print(f"Config loaded: {configs}")

    _demo(configs)