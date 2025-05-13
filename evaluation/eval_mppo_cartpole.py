import sys
sys.path.append("..")

import yaml
import torch
import argparse
import numpy as np
import gymnasium as gym

from tqdm import tqdm

from bc_ppo.ppo_cartpole import PPO


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
    device = 'cpu'
    
    model = PPO(4, 2, 64).to(device)
    model.load_state_dict(torch.load(configs['model']))

    return model


if __name__ == "__main__":
    args = _get_arguments()
    configs = _load_configs(args.config)

    print(f"Config loaded: {configs}")

    device = 'cpu'

    env = gym.make("CartPole-v1")
    
    model = _load_model(configs)
    torch.manual_seed(47)

    episodes = 0
    rewards_all = []

    bar = tqdm(range(configs['num_episodes']))

    mean = np.array([-6.3883489e-01, -1.9440360e-02,
                     -1.2499564e-04, 1.2973925e-03])
    std = np.array([0.5521336, 0.4348506, 0.05661277, 0.30561897])

    obs, _ = env.reset(seed=47)
    while True:
        obs = (obs - mean) / (std + 1e-8)
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
        action, log_prob = model.act(obs_tensor)
        obs, rewards, dones, truncated, info = env.step(action.item())
        # env_vec.render("human")

        rewards_all.append(rewards)

        if dones or truncated:
            obs, _  = env.reset()
            episodes += 1
            bar.update(1)
            if episodes == configs['num_episodes']:
                break
 
    print("Avg reward", np.sum(rewards_all) / configs['num_episodes'])
