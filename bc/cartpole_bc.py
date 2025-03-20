import os
import yaml
import torch
import pickle
import argparse
import numpy as np
import  torch.nn.functional as F

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


def _load_demonstractions(configs):
    with open(configs["demonstrations_path"], 'rb') as f:
        demonstrations = pickle.load(f)

    demonstrations = demonstrations[:min(len(demonstrations),
                                         configs["num_samples"])]

    observations = np.asarray([item[0] for item in demonstrations])
    actions = np.asarray([item[2] for item in demonstrations])
    
    # Compute mean and std
    obs_mean = observations.mean(axis=0)
    obs_std = observations.std(axis=0) + 1e-8  # Avoid division by zero

    print(f"Mean: {obs_mean}, Std: {obs_std}")

    # Normalize observations
    observations = (observations - obs_mean) / obs_std

    bc_dataset = BCDataset(observations, actions)

    return bc_dataset


def _build_model():
    model = MLP_BC(4, 2, hidden_dim=256)

    return model


def _train(configs):
    dataset = _load_demonstractions(configs)
    print(f"Dataset loaded: {dataset.__len__()}")
    dataloader = DataLoader(dataset, batch_size=configs['batch_size'], shuffle=True)
    model = _build_model()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss = torch.nn.CrossEntropyLoss()

    for epoch in range(configs['num_epochs']):
        loss_epoch = []
        correct = 0
        total = 0
        for batch in dataloader:
            obs, action = batch
            obs = torch.tensor(obs, dtype=torch.float32)
            action = torch.tensor(action, dtype=torch.long)

            pred = model(obs)

            # print(pred.shape, action.shape)
            # print(pred, action)
            loss_value = loss(pred, action)

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            loss_epoch.append(loss_value.item())

            predicted_labels = torch.argmax(pred, dim=1)  # Get predicted class
            correct += (predicted_labels == action).sum().item()
            total += action.size(0)

        loss_value = np.mean(loss_epoch)
        print(f"Epoch: {epoch}, Loss: {loss_value.item()}")

        accuracy = (correct / total) * 100  # Compute accuracy %
        print(f"Epoch: {epoch}, Accuracy: {accuracy:.2f}%")

    torch.save(model.state_dict(), configs["model_save"])


if __name__ == '__main__':
    args = _get_arguments()
    configs = _load_configs(args.config)

    print(f"Config loaded: {configs}")

    _train(configs)