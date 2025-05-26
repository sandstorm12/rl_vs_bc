import sys
sys.path.append("..")

import yaml
import argparse
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from tqdm import tqdm

from bc.lunarlander_bc import MLP_BC


class PPO(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(PPO, self).__init__()
        
        self._policy = MLP_BC(input_dim, output_dim)
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


class PPOBuffer:
    def __init__(self):
        self._episodes = [[]]
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        self.advantages = []
        self.returns = []

        self._rewards_mean = 0
        self._rewards_std = 1
        self._advantage_mean = 0
        self._advantage_std = 1

    def store(self, state, action, reward, log_prob, value, done, truncated):
        self._episodes[-1].append((
            state, action, reward, log_prob,
            value, done, truncated))
        
        # Also store in flat arrays for easier batch processing
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done or truncated)
        
        if done or truncated:
            self._episodes.append([])

    def compute_advantages(self, gamma=0.99, lam=0.95, device='cpu'):
        self.advantages = []
        self.returns = []
        
        # Convert to tensors for easier processing
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=device)
        values = torch.tensor(self.values, dtype=torch.float32, device=device)
        dones = torch.tensor(self.dones, dtype=torch.float32, device=device)
        
        # Append a value of 0 for terminal states
        next_values = torch.cat([values[1:], torch.zeros(1, device=device)])
        
        # GAE calculation
        deltas = rewards + gamma * next_values * (1 - dones) - values
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                last_gae = 0
            last_gae = deltas[t] + gamma * lam * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
        
        # Calculate returns (for value function target)
        returns = advantages + values

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        self.advantages = advantages
        self.returns = returns

    def get_minibatch(self, batch_size, device):
        batch_indices = np.random.choice(len(self.states), batch_size, replace=False)
        
        batch_states = torch.stack([self.states[i] for i in batch_indices]).to(device)
        batch_actions = torch.stack([self.actions[i] for i in batch_indices]).to(device)
        batch_log_probs = torch.stack([self.log_probs[i] for i in batch_indices]).to(device)
        batch_advantages = self.advantages[batch_indices].to(device)
        batch_returns = self.returns[batch_indices].to(device)
        
        return batch_states, batch_actions, batch_log_probs, \
            batch_advantages, batch_returns

    def clear(self):
        self._episodes = [[]]
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        self.advantages = []
        self.returns = []


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


def fill_buffer(buffer, ppo, env, device, configs):
    obs, _ = env.reset(seed=rng.randint(1e6))
    mean = np.array([
        0.09450758, 0.59836125, 0.02613196, -0.12162905, 
        0.01558024, 0.0031787, 0.1349, 0.1238
    ])

    std = np.array([
        0.19597459, 0.46255904, 0.13085105, 0.08376966, 
        0.10467913, 0.12128206, 0.34161037, 0.32934433
    ])
  
    for step in range(configs['buffer_size']):
        obs = (obs - mean) / (std + 1e-8)
        obs_tensor = torch.from_numpy(obs).float().to(device)

        with torch.no_grad():
            action, log_prob = ppo.act(obs_tensor, generator=torch_rng)
        
        obs_new, reward, done, truncated, _ = env.step(action.item())

        with torch.no_grad():
            _, _, value = ppo.evaluate(obs_tensor, action)
        
        buffer.store(obs_tensor, action, reward,
                     log_prob.detach(), value.item(),
                     done, truncated)
        obs = obs_new
        if done or truncated:
            obs, _ = env.reset(seed=rng.randint(1e6))

    return configs['buffer_size']


def update(buffer, ppo, optimizer, progress, device, configs):
    buffer.compute_advantages(gamma=0.99, lam=0.95, device=device)
    
    batch_size = configs['batch_size']
    num_epochs = configs['epochs_inner']
    
    clip_param = 0.2
    value_weight = .5
    entropy_weight = (1 - progress) * .01
    total_loss = 0
    total_surrogate_loss = 0
    total_value_loss = 0
    total_entropy_loss = 0
    
    for _ in range(num_epochs):
        n_samples = len(buffer.states)
        n_batches = n_samples // batch_size
        
        for _ in range(n_batches):
            states, actions, old_log_probs, advantages, returns = \
                buffer.get_minibatch(batch_size, device)

            new_log_probs, entropies, values = \
                ppo.evaluate(states, actions)
            values = values.squeeze()
            
            # Compute policy loss
            ratios = torch.exp(new_log_probs - old_log_probs)
            clipped_ratios = torch.clamp(ratios, 1 - clip_param, 1 + clip_param)
            surrogate_loss = -torch.min(ratios * advantages, clipped_ratios * advantages).mean()
            
            # Compute value loss
            value_loss = F.mse_loss(values, returns)
            value_loss /= returns.var() + 1e-8
            
            # Compute entropy loss
            entropy_loss = entropies.mean()
            
            # Total loss
            loss = surrogate_loss + \
                value_weight * value_loss - \
                entropy_weight * entropy_loss
            
            # Perform update
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ppo.parameters(), max_norm=0.5)
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            total_surrogate_loss += surrogate_loss.item()
            total_value_loss += value_weight * value_loss.item()
            total_entropy_loss += entropy_weight * 1. * entropy_loss.item()
    
    # Compute average over all mini-batch updates
    total_updates = num_epochs * n_batches
    avg_loss = total_loss / total_updates
    avg_surrogate_loss = total_surrogate_loss / total_updates
    avg_value_loss = total_value_loss / total_updates
    avg_entropy_loss = total_entropy_loss / total_updates
    
    # Calculate average episode reward (from buffer episodes data)
    rewards_total = 0.0
    num_episodes = 0
    
    for episode in buffer._episodes:
        if len(episode) < 2:
            continue
        episode_reward = sum(item[2] for item in episode)  # item[2] is the reward
        rewards_total += episode_reward
        num_episodes += 1
    
    avg_reward = rewards_total / max(1, num_episodes)  # Avoid division by zero
    
    return avg_loss, avg_reward, avg_surrogate_loss, \
        avg_value_loss, avg_entropy_loss


def train(ppo, env, device, configs):
    buffer = PPOBuffer()
    optimizer = torch.optim.Adam([
        {'params': ppo._policy.parameters(), 'lr': 1e-4},
        {'params': ppo._value.parameters(), 'lr': 1e-3}
    ])

    step_size = configs['train_steps'] / configs['buffer_size'] // 20
    scheduler = StepLR(optimizer, step_size=step_size, gamma=0.9)

    avg_loss_history = []
    avg_reward_history = []
    avg_surrogate_loss_history = []
    avg_value_loss_history = []
    avg_entropy_loss_history = []

    total_timesteps = configs['train_steps']
    timesteps = 0
    bar = tqdm(range(total_timesteps))
    while timesteps < total_timesteps:
        progress = min(timesteps / total_timesteps, 1.0)

        steps = fill_buffer(buffer, ppo, env, device, configs)
        timesteps += steps
        avg_loss, avg_reward, avg_surrogate_loss, \
            avg_value_loss, avg_entropy_loss = \
            update(buffer, ppo, optimizer, progress, device, configs)

        avg_loss_history.append(avg_loss)
        avg_reward_history.append(avg_reward)
        avg_surrogate_loss_history.append(avg_surrogate_loss)
        avg_value_loss_history.append(avg_value_loss)
        avg_entropy_loss_history.append(avg_entropy_loss)

        bar.update(steps)
        bar.set_description(
            'AL %.3f AR: %.1f ASL %.3f AVL %.3f AEL %.3f' % \
            (avg_loss, avg_reward, avg_surrogate_loss,
             avg_value_loss, avg_entropy_loss))
        bar.refresh()
        buffer.clear()

        scheduler.step()
        for param_group in optimizer.param_groups:
            print("Current LR:", param_group['lr'])

    plt.plot(avg_loss_history, label='Average Loss')
    plt.plot(avg_reward_history, label='Average Reward')
    plt.plot(avg_surrogate_loss_history, label='Average Surrogate Loss')
    plt.plot(avg_value_loss_history, label='Average Value Loss')
    plt.plot(avg_entropy_loss_history, label='Average Entropy Loss')
    plt.legend()
    plt.show()

    torch.save(ppo.state_dict(), configs['save_path'])


if __name__ == '__main__':
    args = _get_arguments()
    configs = _load_configs(args.config)

    print(f"Config loaded: {configs}")

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'

    # Seeding
    rng = np.random.RandomState(47)
    torch_rng = torch.Generator().manual_seed(47)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(47)
    np.random.seed(47)

    ppo = PPO(8, 4, 64).to(device)

    env = gym.make("LunarLander-v2", continuous=False,
                   render_mode="rgb_array")
    env.reset(seed=rng.randint(1e6))
    env.action_space.seed(47)
    env.observation_space.seed(47)

    train(ppo, env, device, configs)
