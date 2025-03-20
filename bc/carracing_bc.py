import yaml
import torch
import shelve
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


class CNN_BC(nn.Module):
    def __init__(self, width=96, height=96, input_channels=3, num_actions=5):
        super(CNN_BC, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)  # (C, H, W) -> (32, H, W)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # (64, H, W)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # (128, H, W)

        self.pool = nn.MaxPool2d(2, 2)  # Reduces spatial dimensions by half
        self.fc1 = nn.Linear(128 * (height//8) * (width//8), 256)  # Flattened size
        self.fc2 = nn.Linear(256, num_actions)  # Output action logits

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # First Conv + Pool
        x = self.pool(F.relu(self.conv2(x)))  # Second Conv + Pool
        x = self.pool(F.relu(self.conv3(x)))  # Third Conv + Pool

        x = torch.flatten(x, start_dim=1)  # Flatten before FC layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # No activation, goes into CrossEntropyLoss

        return x
    

class BCDataset(Dataset):
    def __init__(self, database_path, max_size=None):
        self._database_path = database_path
        self._database = shelve.open(self._database_path, 'r')

        self._keys = list(self._database.keys())

        if max_size is not None:
            self._keys = self._keys[:max_size]

    def __len__(self):
        return len(self._keys)
    
    def __getitem__(self, idx):
        demo = self._database[self._keys[idx]]

        obs, _, action, _, _, _, _ = demo

        obs = torch.tensor(obs, dtype=torch.float32)
        obs = obs.permute(2, 0, 1)

        action = torch.tensor(action, dtype=torch.long)

        return obs, action


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


def _load_demonstractions(configs):
    bc_dataset = BCDataset(configs['demonstrations_path'],
                           configs['num_samples'])

    return bc_dataset


def _build_model():
    model = CNN_BC()

    return model


def _train(configs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = _load_demonstractions(configs)
    print(f"Dataset loaded: {dataset.__len__()}")
    dataloader = DataLoader(dataset, batch_size=configs['batch_size'],
                            shuffle=True, num_workers=8)
    print(f"Dataloader created")
    model = _build_model()
    model.to(device)
    print(f"Model built")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,
                                 weight_decay=1e-5)
    loss = torch.nn.CrossEntropyLoss()

    for epoch in range(configs['num_epochs']):
        loss_epoch = []
        correct = 0
        total = 0
        for batch in dataloader:
            obs, action = batch
            obs = obs.to(device)
            action = action.to(device)

            pred = model(obs)

            loss_value = loss(pred, action)

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            loss_epoch.append(loss_value.detach().cpu().item())

            predicted_labels = torch.argmax(pred, dim=1)
            correct += (predicted_labels == action).sum().item()
            total += action.size(0)

        loss_value = np.mean(loss_epoch)
        print(f"Epoch: {epoch}, Loss: {loss_value}")

        accuracy = (correct / total) * 100  # Compute accuracy %
        print(f"Epoch: {epoch}, Accuracy: {accuracy:.2f}%")

    torch.save(model.state_dict(), configs["model_save"])


if __name__ == '__main__':
    args = _get_arguments()
    configs = _load_configs(args.config)

    print(f"Config loaded: {configs}")

    _train(configs)
