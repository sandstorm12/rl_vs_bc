import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np


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

def run_cartpole(weights_path, num_episodes=5, hidden_dim=64):
    # Create CartPole environment with human rendering
    env = gym.make('LunarLander-v2', render_mode='human')
    
    # Get dimensions
    input_dim = env.observation_space.shape[0]  # 8 for CartPole
    output_dim = env.action_space.n            # 4 for CartPole
    
    # Initialize model
    model = PPO(input_dim, output_dim, hidden_dim)
    
    # Load pre-trained weights
    try:
        model.load_state_dict(torch.load(weights_path))
        print(f"Successfully loaded weights from {weights_path}")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return
    
    # Set model to evaluation mode
    model.eval()
    
    # Run episodes
    for episode in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):  # Handle case where reset returns (state, info)
            state = state[0]
        
        done = False
        total_reward = 0
        
        while not done:
            # Convert state to tensor
            mean = np.array([
                0.09450758, 0.59836125, 0.02613196, -0.12162905, 
                0.01558024, 0.0031787, 0.1349, 0.1238
            ])

            std = np.array([
                0.19597459, 0.46255904, 0.13085105, 0.08376966, 
                0.10467913, 0.12128206, 0.34161037, 0.32934433
            ])
            state = (state - mean) / std
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Get action
            with torch.no_grad():
                action, _ = model.act(state_tensor)
            action = action.item()
            
            # Step environment
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated  # Handle both termination conditions
            
            total_reward += reward
            state = next_state
            
            # Render the environment
            env.render()
        
        print(f"Episode {episode + 1}/{num_episodes} - Total Reward: {total_reward}")


if __name__ == "__main__":
    # Specify the path to your pre-trained weights file
    weights_path = "./weights/lunarlander.pth"  # Replace with your actual weights file path
    
    # Run the model
    run_cartpole(weights_path, num_episodes=5, hidden_dim=64)