import torchvision
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

from typing import List

from data_loader.Datafetcher import Datafetcher

class MnistDataset(Dataset):
	def __init__(self: "MnistDataset", file_save_path: str = "./data/", train: bool = True, transform=None) -> None:
		train_dataset = torchvision.datasets.MNIST(root=file_save_path, train=True, download=True)
		test_dataset = torchvision.datasets.MNIST(root=file_save_path, train=False, download=True)

		if train:
			self.data = train_dataset.data.div(255)
		else:
			self.data = test_dataset.data.div(255)
		if transform:
			self.data = transform(self.data)

	def __len__(self: "MnistDataset"):
		return len(self.data)

	def __getitem__(self: "MnistDataset", index: List[int]) -> np.ndarray:
		if torch.is_tensor(index):
			index = index.tolist()

		sample = torch.Tensor(self.data[index]).float()
		return sample		

class MnistDatafetcher(Datafetcher):
	def __init__(self: "MnistDatafetcher", file_save_path: str = "./data/", transform=None) -> None:
		self.train_dataset = MnistDataset(file_save_path, train=True, transform=transform)
		self.test_dataset = MnistDataset(file_save_path, train=False, transform=transform)

	def get_train_dataset(self: "MnistDatafetcher") -> Dataset:
		return self.train_dataset

	def get_test_dataset(self: "MnistDatafetcher") -> Dataset:
		return self.test_dataset

	def get_train_dataloader(self: "MnistDatafetcher", batch_size: int = 32) -> DataLoader:
		return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

	def get_test_dataloader(self: "MnistDatafetcher", batch_size: int = 32) -> DataLoader:
		return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

	def num_features(self: "MnistDatafetcher") -> int:
		return self.train_dataset.data.shape[1]
