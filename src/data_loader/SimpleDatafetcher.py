import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader

from data_loader.Datafetcher import Datafetcher

class _SimpleDataset(Dataset):
	def __init__(self: "_SimpleDataset", transform=None) -> None:
		self.data = np.array([
			[0, 0, 0, 0],
			[0, 0, 0, 1], 
			[0, 0, 1, 0],
			[0, 0, 1, 1], 
			[0, 1, 0, 0],
			[0, 1, 0, 1], 
			[0, 1, 1, 0],
			[0, 1, 1, 1], 
			[1, 0, 0, 0],
			[1, 0, 0, 1], 
			[1, 0, 1, 0],
			[1, 0, 1, 1], 
			[1, 1, 0, 0],
			[1, 1, 0, 1], 
			[1, 1, 1, 0],
			[1, 1, 1, 1], 
		], dtype=np.float32)

		self.data = np.array([val for val in self.data for _ in (range(1000))])

		if transform:
			self.data = transform(self.data)
	
	def __len__(self: "_SimpleDataset") -> int:
		return len(self.data[0])

	def __getitem__(self, index) -> np.ndarray:
		return self.data[0][index], self.data[0][index]

class SimpleDatafetcher(Datafetcher):
	def __init__(self: "SimpleDatafetcher", transform=None) -> None:
		if transform:
			transformations = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), transform])
		else:
			transformations = torchvision.transforms.ToTensor()
		
		self.train_data = _SimpleDataset(transform=transformations)
		self.test_data = _SimpleDataset(transform=transformations)

	def get_train_dataset(self: "SimpleDatafetcher") -> _SimpleDataset:
		return self.train_data
		
	def get_test_dataset(self: "SimpleDatafetcher") -> _SimpleDataset:
		return self.test_data

	def get_train_dataloader(self: "SimpleDatafetcher", batch_size: int = 32) -> DataLoader:
		return DataLoader(self.train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
	
	def get_test_dataloader(self: "SimpleDatafetcher", batch_size: int = 32) -> DataLoader:
		return DataLoader(self.test_data, batch_size=batch_size, shuffle=False)