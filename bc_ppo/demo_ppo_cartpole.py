import yaml
import argparse
import numpy as np
import gymnasium as gym

import torch
import torch.nn as nn


class PPO(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(PPO, self).__init__()
        
        self._backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self._policy = nn.Linear(hidden_dim, output_dim)
        self._value = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self._backbone(x)
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
    env = gym.make('CartPole-v1', render_mode='human')
    
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
    
    for episode in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        
        done = False
        total_reward = 0
        
        while not done:
            mean = np.array([-6.3883489e-01, -1.9440360e-02,
                             -1.2499564e-04, 1.2973925e-03])
            std = np.array([0.5521336, 0.4348506,
                            0.05661277, 0.30561897])
            state = (state - mean) / std
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            with torch.no_grad():
                action, _ = model.act(state_tensor)
            action = action.item()
            
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            
            total_reward += reward
            state = next_state
            
            env.render()
        
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"Total Reward: {total_reward}")


if __name__ == "__main__":
    args = _get_arguments()
    configs = _load_configs(args.config)

    print(f"Config loaded: {configs}")

    run_cartpole(configs['model'], num_episodes=5, hidden_dim=64)
