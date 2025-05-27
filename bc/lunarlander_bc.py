import yaml
import torch
import pickle
import argparse
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm

from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader


class MLP_BC(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(MLP_BC, self).__init__()
        self.layer1 = torch.nn.Linear(input_dim, hidden_dim)
        self.layer2 = torch.nn.Linear(hidden_dim, hidden_dim)
        # self.layer3 = torch.nn.Linear(hidden_dim, hidden_dim)
        # self.layer4 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.layer5 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        # x = F.relu(self.layer3(x))
        # x = F.relu(self.layer4(x))
        x = self.layer5(x)
        
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

    print(f"Demonstractions loaded: {len(demonstrations)}")

    demonstrations = demonstrations[:min(len(demonstrations),
                                         configs["num_samples"])]

    observations = np.asarray([item[0] for item in demonstrations])
    actions = np.asarray([item[2] for item in demonstrations])
    
    # Compute mean and std
    obs_mean = observations.mean(axis=0)
    obs_std = observations.std(axis=0) + 1e-8

    print(f"Mean: {obs_mean}, Std: {obs_std}")

    observations = (observations - obs_mean) / obs_std

    bc_dataset = BCDataset(observations, actions)

    return bc_dataset


def _build_model():
    model = MLP_BC(8, 4, hidden_dim=512)

    return model


def _train(configs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'

    dataset = _load_demonstractions(configs)
    print(f"Dataset loaded: {dataset.__len__()}")
    
    generator = torch.Generator().manual_seed(42)
    dataset_train, dataset_test = torch.utils.data.random_split(
        dataset,
        [int(len(dataset) * 0.8),
         len(dataset) - int(len(dataset) * 0.8)],
         generator=generator)
    print(f"Dataset train loaded: {dataset_train.__len__()}")
    print(f"Dataset test loaded: {dataset_test.__len__()}")
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=configs['batch_size'],
        shuffle=True)
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=configs['batch_size'],
        shuffle=True)
    
    model = _build_model()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # scheduler = StepLR(optimizer,
    #                    step_size=configs['num_epochs'] / 4,
    #                    gamma=0.1)
    loss = torch.nn.CrossEntropyLoss()

    bar = tqdm(range(configs['num_epochs']))
    for epoch in bar:
        loss_epoch = []
        correct_train = 0
        total_train = 0
        for batch in dataloader_train:
            obs, action = batch
            obs = torch.tensor(obs, dtype=torch.float32).to(device)
            action = torch.tensor(action, dtype=torch.long).to(device)

            pred = model(obs)

            # print(pred.shape, action.shape)
            # print(pred, action)
            loss_value = loss(pred, action)

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            loss_epoch.append(loss_value.item())

            predicted_labels = torch.argmax(pred, dim=1)
            # print(predicted_labels, action)
            correct_train += (predicted_labels == action).sum().item()
            total_train += action.size(0)

        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for batch in dataloader_test:
                obs, action = batch
                obs = torch.tensor(obs, dtype=torch.float32).to(device)
                action = torch.tensor(action, dtype=torch.long).to(device)

                pred = model(obs)

                predicted_labels = torch.argmax(pred, dim=1)
                # print(predicted_labels, action)
                correct_test += (predicted_labels == action).sum().item()
                total_test += action.size(0)

        loss_value = np.mean(loss_epoch)
        # print(f"Epoch: {epoch}, Loss: {loss_value.item()}")

        accuracy_train = (correct_train / total_train) * 100  # Compute accuracy %
        accuracy_test = (correct_test / total_test) * 100  # Compute accuracy %
        
        # print(f"Epoch: {epoch}, Accuracy: {accuracy:.2f}%")

        bar.set_description(f"L: {loss_value:.4f}, "
                            f"ATr: {accuracy_train:.2f}%, "
                            f"ATe: {accuracy_test:.2f}%")
        
        # scheduler.step()
        # print(f"Epoch {epoch+1}, LR: {scheduler.get_last_lr()[0]}")

    torch.save(model.state_dict(), configs["model_save"])


if __name__ == '__main__':
    args = _get_arguments()
    configs = _load_configs(args.config)

    print(f"Config loaded: {configs}")

    torch.manual_seed(47)
    np.random.seed(47)

    _train(configs)
