import torch
import torchvision
import numpy as np
import pandas as pd
from typing import Tuple
from torch.utils.data import DataLoader

from data.WineQualityDataset import WineQualityDataset


def get_mnist_dataloaders(batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
	transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
	train_dataset = torchvision.datasets.MNIST(root="./datasets/", train=True, transform=transform, download=True)
	test_dataset = torchvision.datasets.MNIST(root="./datasets/", train=False, transform=transform, download=True)

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
	
	return train_loader, test_loader

def get_mnist_dataset() -> Tuple[torch.Tensor, torch.Tensor]:
	transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
	train_dataset = torchvision.datasets.MNIST(root="./datasets/", train=True, transform=transform, download=True).data.float()
	test_dataset = torchvision.datasets.MNIST(root="./datasets/", train=False, transform=transform, download=True).data.float()

	#torchvision.transforms.Normalize((0.1307,), (0.3081,))
	train_dataset = torch.nn.functional.normalize(train_dataset)
	test_dataset = torch.nn.functional.normalize(test_dataset)
	return train_dataset, test_dataset

def simple_test_dataset() -> Tuple[torch.Tensor, torch.Tensor]:
	train = torch.Tensor([1, 0, 1, 0, 1, 0, 1, 0])
	return train, train

def get_winequality_dataloader(batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
	train_dataset = WineQualityDataset("./datasets/WineQuality/winequality-white.csv", train=True)
	test_dataset = WineQualityDataset("./datasets/WineQuality/winequality-white.csv", test=True)

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

	return train_loader, test_loader



