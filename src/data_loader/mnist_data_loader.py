import torch
import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler

from typing import List, Tuple

scaler = MinMaxScaler()

class _MnistDataset(Dataset):
    def __init__(self: "_MnistDataset", train: bool = True) -> None:
        train_dataset = MNIST('./data/', train=True, download=True)
        test_dataset = MNIST('./data/', train=False, download=True)

        train_data: torch.Tensor = torch.flatten(train_dataset.data, start_dim=1)
        test_data: torch.Tensor = torch.flatten(test_dataset.data, start_dim=1)

        scaler.fit(torch.cat((train_data, test_data)))

        transformed_train_data: np.ndarray= scaler.transform(train_data)
        transformed_test_data: np.ndarray = scaler.transform(test_data)

        if train:
            self.data = torch.Tensor(transformed_train_data)
        else:
            self.data = torch.Tensor(transformed_test_data)

    def __len__(self: "_MnistDataset") -> int:
        return len(self.data)

    def __getitem__(self: "_MnistDataset", index: List[int]) -> np.ndarray:
        if torch.is_tensor(index):
            index = index.tolist()

        sample = self.data[index]

        return sample, sample	



def mnist_loaders(train_batch_size=32, test_batch_size=32) -> Tuple[DataLoader, DataLoader]:
    train_dataset = _MnistDataset(train=True)
    test_dataset = _MnistDataset(train=False)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
 
    return train_loader, test_loader

    #transform = Compose([
    #    ToTensor(),
    #    Normalize((0.1307,), (0.3081,)),
    #    Lambda(lambda x: torch.flatten(x))])

    #train_loader = DataLoader(
    #    MNIST('./data/', train=True,
    #          download=True,
    #          transform=transform),
    #    batch_size=train_batch_size, shuffle=True)

    #test_loader = DataLoader(
    #    MNIST('./data/', train=False,
    #          download=True,
    #          transform=transform),
    #    batch_size=test_batch_size, shuffle=False)
    #return train_loader, test_loader