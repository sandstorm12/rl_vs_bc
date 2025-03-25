import time
import numpy as np
import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor
from stable_baselines3.common.vec_env import DummyVecEnv


import sys
sys.path.append("..")

from bc.carracing_bc import CNN_BC


class PPO(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(PPO, self).__init__()
        
        self._policy = CNN_BC()

        self._value = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(128 * (input_dim//8) * (input_dim//8), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
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
        bc_model = CNN_BC()  # Initialize a CNN_BC model with the same architecture
        bc_model.load_state_dict(torch.load(path))  # Load trained weights
        bc_model.eval()  # Set to evaluation mode

        # Copy weights from CNN_BC to PPO's policy network
        self._policy.load_state_dict(bc_model.state_dict())


    def freeze_policy(self, freeze=True):
        for param in self._policy.parameters():
            param.requires_grad = not freeze

    
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
        
        # # Normalize advantages
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        self.advantages = advantages
        self.returns = returns
    
    def get_minibatch(self, batch_size, device):
        batch_indices = np.random.choice(len(self.states), batch_size, replace=False)
        
        batch_states = torch.stack([self.states[i] for i in batch_indices]).to(device)
        batch_actions = torch.stack([self.actions[i] for i in batch_indices]).to(device)
        batch_log_probs = torch.stack([self.log_probs[i] for i in batch_indices]).to(device)
        batch_advantages = self.advantages[batch_indices].to(device)
        batch_returns = self.returns[batch_indices].to(device)
        
        return batch_states, batch_actions, batch_log_probs, batch_advantages, batch_returns
    
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

def fill_buffer(buffer, ppo, envs, device):
    if hasattr(fill_buffer, 'obs'):
        obs = fill_buffer.obs
    else:
        obs = envs.reset()
    
    for step in range(512 // envs.num_envs):
        obs_tensor = torch.from_numpy(obs).float().to(device).permute(0, 3, 1, 2)
        obs_tensor.div_(255.0),

        with torch.no_grad():
            action, log_prob = ppo.act(obs_tensor)
            obs_new, reward, done, info = envs.step(action.cpu().numpy())
            _, _, value = ppo.evaluate(obs_tensor, action)

        for idx in range(envs.num_envs):
            obs_tensor_env = obs_tensor[idx]
            action_env = action[idx]
            log_prob_env = log_prob[idx]
            value_env = value[idx]
            reward_env = reward[idx]
            done_env = done[idx]
            truncated = info[idx].get('TimeLimit.truncated', False)

            buffer.store(obs_tensor_env, action_env, reward_env,
                         log_prob_env.detach(), value_env,
                         done_env, truncated)

            if done_env or truncated:
                obs_new[idx], _ = envs.envs[idx].reset()

        obs = obs_new


def update(buffer, ppo, optimizer, progress, device):
    # Compute advantages once before training
    buffer.compute_advantages(gamma=0.99, lam=0.95, device=device)
    
    batch_size = 512  
    num_epochs = 2
    clip_param = 0.2
    entropy_weight = 0
    value_weight = .1

    # Get total number of samples once
    n_samples = len(buffer.states)
    n_batches = n_samples // batch_size

    # Initialize loss trackers
    avg_loss = 0
    avg_surrogate_loss = 0
    avg_value_loss = 0
    avg_entropy_loss = 0

    for _ in range(num_epochs):
        for _ in range(n_batches):
            # Sample a minibatch
            states, actions, old_log_probs, advantages, returns = \
                buffer.get_minibatch(batch_size, device)
            
            # **Batch evaluation to avoid looping**
            new_log_probs, entropies, values = ppo.evaluate(states, actions)

            # Compute policy loss
            ratios = torch.exp(new_log_probs - old_log_probs)
            clipped_ratios = torch.clamp(ratios, 1 - clip_param, 1 + clip_param)
            surrogate_loss = -torch.min(ratios * advantages, clipped_ratios * advantages).mean()

            # Compute value loss
            value_loss = F.mse_loss(values.squeeze(), returns)

            # Compute entropy loss
            entropy_loss = entropies.mean()

            # Total loss
            loss = surrogate_loss + value_weight * value_loss - entropy_weight * entropy_loss

            # Perform update
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ppo.parameters(), max_norm=0.5)
            optimizer.step()

            # Update running averages
            avg_loss += loss.item()
            avg_surrogate_loss += surrogate_loss.item()
            avg_value_loss += value_weight * value_loss.item()
            avg_entropy_loss += entropy_weight * entropy_loss.item()

    # Compute final averages
    total_updates = num_epochs * n_batches
    avg_loss /= total_updates
    avg_surrogate_loss /= total_updates
    avg_value_loss /= total_updates
    avg_entropy_loss /= total_updates

    # **Efficient reward calculation**
    avg_reward = sum(sum(item[2] for item in ep) for ep in buffer._episodes if len(ep) > 1) / max(1, len(buffer._episodes))

    return avg_loss, avg_reward, avg_surrogate_loss, avg_value_loss, avg_entropy_loss


def train(ppo, env, device):
    buffer_current = PPOBuffer()
    buffer_reserve = PPOBuffer()
    optimizer = torch.optim.Adam([
        {'params': ppo._policy.parameters(), 'lr': 1e-4},
        {'params': ppo._value.parameters(), 'lr': 1e-3},
    ])

    start = time.time()
    fill_buffer(buffer_current, ppo, env, device)
    print("First buffer filled", time.time() - start)

    ppo.freeze_policy(True)
    
    EPOCHS = 1000
    EPOCHS_VALUE = 50
    bar = tqdm(range(EPOCHS))
    for epoch in bar:
        progress = epoch / EPOCHS

        if epoch % EPOCHS_VALUE == EPOCHS_VALUE - 1:
            ppo.freeze_policy(False)
            print("Training Policy")

        avg_loss, avg_reward, avg_surrogate_loss, \
            avg_value_loss, avg_entropy_loss = \
                update(buffer_current, ppo, optimizer, progress, device)
        fill_buffer(buffer_reserve, ppo, env, device)

        bar.set_description('Loss %.3f Rewards: %.1f s_loss %.3f v_loss %.3f e_loss %.3f' % ( \
            avg_loss, avg_reward, avg_surrogate_loss, avg_value_loss, avg_entropy_loss))

        buffer_current.clear()
        
        buffer_current, buffer_reserve = buffer_reserve, buffer_current

        torch.save(ppo.state_dict(), './weights/carracing.pth')


def _create_env(num_envs=4):
    def make_env():
        return gym.make("CarRacing-v2", continuous=False, render_mode="rgb_array")

    envs = DummyVecEnv(
        [make_env
         for _ in range(num_envs)]
    )

    return envs


# Just for test
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'

    ppo = PPO(input_dim=96, output_dim=5, hidden_dim=256).to(device)
    ppo.load_policy_bc('/home/hamid/Documents/indie_projects/rl_vs_bc/bc/weights/carracing_bc') 

    envs = _create_env()
    print("Num envs", envs.num_envs)

    train(ppo, envs, device)

    torch.save(ppo.state_dict(), './weights/carracing.pth')