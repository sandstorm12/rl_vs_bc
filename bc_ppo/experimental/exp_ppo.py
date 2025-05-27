import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from tqdm import tqdm

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
        
        # Normalize advantages
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

def fill_buffer(buffer, ppo, env, device):
    obs, _ = env.reset()
    mean = np.array([-6.3883489e-01, -1.9440360e-02, -1.2499564e-04, 1.2973925e-03])
    std = np.array([0.5521336, 0.4348506, 0.05661277, 0.30561897])
        
    for step in range(2048):
        obs = (obs - mean) / (std + 1e-8)
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
        action, log_prob = ppo.act(obs_tensor)
        obs_new, reward, done, truncated, info = env.step(action.item())
        _, _, value = ppo.evaluate(obs_tensor, action)
        buffer.store(obs_tensor, action, reward,
                     log_prob.detach(), value.item(),
                     done, truncated)
        obs = obs_new
        if done or truncated:
            obs, _ = env.reset()

def update(buffer, ppo, optimizer, progress, device):
    # First compute advantages for all samples
    buffer.compute_advantages(gamma=0.99, lam=0.95, device=device)
    
    batch_size = 64  # Mini-batch size
    num_epochs = 10  # Number of passes through the data
    
    clip_param = 0.2
    entropy_weight = (1 - progress) * 0.01
    total_loss = 0
    total_surrogate_loss = 0
    total_value_loss = 0
    total_entropy_loss = 0
    
    for _ in range(num_epochs):
        # Get total number of samples
        n_samples = len(buffer.states)
        # How many mini-batches we'll make
        n_batches = n_samples // batch_size
        
        for _ in range(n_batches):
            # Sample a minibatch
            states, actions, old_log_probs, advantages, returns = buffer.get_minibatch(batch_size, device)
            
            # Current policy evaluation
            new_log_probs, entropies, values = zip(*[ppo.evaluate(s, a) for s, a in zip(states, actions)])
            new_log_probs = torch.stack(new_log_probs)
            # entropies = torch.stack(entropies)
            values = torch.stack(values).squeeze()
            
            # Compute policy loss
            ratios = torch.exp(new_log_probs - old_log_probs)
            clipped_ratios = torch.clamp(ratios, 1 - clip_param, 1 + clip_param)
            surrogate_loss = -torch.min(ratios * advantages, clipped_ratios * advantages).mean()
            
            # Compute value loss
            value_loss = F.mse_loss(values, returns)
            
            # Compute entropy loss
            # entropy_loss = entropies.mean()
            
            # Total loss
            loss = surrogate_loss + 0.5 * value_loss
            
            # Perform update
            optimizer.zero_grad()
            loss.backward()
            # Optional: clip gradients
            torch.nn.utils.clip_grad_norm_(ppo.parameters(), max_norm=0.5)
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            total_surrogate_loss += surrogate_loss.item()
            total_value_loss += value_loss.item()
            # total_entropy_loss += entropy_loss.item()
    
    # Compute average over all mini-batch updates
    total_updates = num_epochs * n_batches
    avg_loss = total_loss / total_updates
    avg_surrogate_loss = total_surrogate_loss / total_updates
    avg_value_loss = total_value_loss / total_updates
    # avg_entropy_loss = total_entropy_loss / total_updates
    
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
    
    return avg_loss, avg_reward

def train(ppo, env, device):
    buffer = PPOBuffer()
    optimizer = torch.optim.Adam(ppo.parameters(), lr=3e-4)
    
    EPOCHS = 200
    bar = tqdm(range(EPOCHS))
    for epoch in bar:
        fill_buffer(buffer, ppo, env, device)
        progress = epoch / EPOCHS
        loss, rewards_total = update(buffer, ppo, optimizer, progress, device)
        bar.set_description('Loss %.3f Rewards: %.1f' % (loss, rewards_total))
        buffer.clear()

# Just for test
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    ppo = PPO(4, 2, 64).to(device)
    env = gym.make("CartPole-v1")
    train(ppo, env, device)
    #save model
    torch.save(ppo.state_dict(), './weights/ppo.pth')