import sys
sys.path.append("..")

import yaml
import argparse
import numpy as np
import gymnasium as gym

import torch
import torch.nn as nn

from tqdm import tqdm
from bc.cartpole_bc import MLP_BC
from stable_baselines3.common.env_util import make_vec_env


class PPO(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(PPO, self).__init__()
        
        self._policy = MLP_BC(4, 2, 256)
        self._value = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, x):
        action_logits = self._policy(x)
        value = self._value(x)
        
        return action_logits, value
    
    def act(self, x):
        action_logits, _ = self.forward(x)
        
        distribution = torch.distributions.Categorical(logits=action_logits)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        
        return action, log_prob
        
    def evaluate(self, x, action):
        action_logits, value = self.forward(x)
        distribution = torch.distributions.Categorical(logits=action_logits)
        log_prob = distribution.log_prob(action)
        entropy = distribution.entropy()
        
        return log_prob, entropy, value
    
    def load_policy_bc(self, path):
        bc_model = MLP_BC(4, 2, 256)
        bc_model.load_state_dict(torch.load(path))
        bc_model.eval()

        self._policy.load_state_dict(bc_model.state_dict())

    def freeze_policy(self, freeze=True):
        for param in self._policy.parameters():
            param.requires_grad = not freeze


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


def run_cartpole(weights_path, num_episodes=5, hidden_dim=64):
    env = make_vec_env("CartPole-v1", n_envs=1, seed=42)
    
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    
    model = PPO(input_dim, output_dim, hidden_dim)
    
    try:
        model.load_state_dict(torch.load(weights_path))
        print(f"Successfully loaded weights from {weights_path}")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return
    
    model.eval()
        
    episodes = 0
    rewards_all = []

    bar = tqdm(range(configs['num_episodes']))
    
    obs = env.reset()
    while True:
        mean = np.array([-6.3883489e-01, -1.9440360e-02,
                            -1.2499564e-04, 1.2973925e-03])
        std = np.array([0.5521336, 0.4348506,
                        0.05661277, 0.30561897])
        obs = (obs - mean) / std
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        
        with torch.no_grad():
            action, _ = model.act(obs_tensor)
        action = action.item()
        
        next_state, rewards, dones, info = env.step([action])
        dones = dones or info[0]['TimeLimit.truncated']
        
        obs = next_state
        
        env.render("human")

        rewards_all.append(rewards)

        if dones:
            obs = env.reset()
            episodes += 1
            bar.update(1)
            if episodes == configs['num_episodes']:
                break
        
    print("Avg reward", np.sum(rewards_all) / configs['num_episodes'])


if __name__ == "__main__":
    args = _get_arguments()
    configs = _load_configs(args.config)

    print(f"Config loaded: {configs}")

    run_cartpole(configs['model'], num_episodes=configs['num_episodes'], hidden_dim=64)
