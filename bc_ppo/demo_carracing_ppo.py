import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np


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

        print(bc_model.state_dict().keys())

        # Copy weights from CNN_BC to PPO's policy network
        self._policy.load_state_dict(bc_model.state_dict())


def run_cartpole(weights_path, num_episodes=5, hidden_dim=256):
    # Create CartPole environment with human rendering
    env = gym.make("CarRacing-v2", continuous=False, render_mode="human")
    
    # Initialize model
    model = PPO(input_dim=96, output_dim=5, hidden_dim=hidden_dim)
    
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
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            state_tensor = state_tensor.permute(0, 3, 1, 2).div(255.0)
            
            # Get action
            with torch.no_grad():
                action, _ = model.act(state_tensor)
            action = action.item()

            print(action)
            
            # Step environment
            next_state, reward, done, truncated, _ = env.step(action)
            done = done  # Handle both termination conditions
            
            total_reward += reward
            state = next_state
            
            # Render the environment
            env.render()
        
        print(f"Episode {episode + 1}/{num_episodes} - Total Reward: {total_reward}")


if __name__ == "__main__":
    # Specify the path to your pre-trained weights file
    weights_path = "./weights/carracing.pth"  # Replace with your actual weights file path
    
    # Run the model
    run_cartpole(weights_path, num_episodes=5, hidden_dim=256)