import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader

from typing import Tuple

class _SimpleDataset(Dataset):
	def __init__(self: "_SimpleDataset", transform=None, train: bool = False) -> None:
		self.data = np.array([
			[0.1, 0.4, 0.3, 0.8, 0.2],
			[0.1, 0, 0, 0.4, 0.2], 
			[0.1, 0.2, 0.3, 0, 0.2],
			[0.1, 0, 0.9, 0.2, 0.2], 
			[0.1, 0.8, 0, 0, 0.3],
			[0.1, 0.7, 0, 0.7, 0.2], 
			[0.1, 0.2, 0.5, 0, 0.2],
			[0.1, 0.1, 0.1, 0.8, 0.3], 
			#[0.6, 0, 0, 0, 0],
			#[0.6, 0, 0, 0.4, 0], 
			#[0.6, 0, 0.5, 0, 0],
			#[0.6, 0, 0.3, 0.3, 0], 
			#[0.6, 0.8, 0, 0, 0],
			#[0.6, 0.9, 0, 1, 0], 
			#[0.6, 1, 0.5, 0, 0],
			#[0.6, 1, 0.6, 0.7, 0], 
		], dtype=np.float32)

		self.train_data = np.array(self.data)

		self.data = np.array([val for val in self.data for _ in (range(200))])

		if not train:
			self.data = self.train_data

		if transform:
			self.data = transform(self.data)
	
	def __len__(self: "_SimpleDataset") -> int:
		return len(self.data[0])

	def __getitem__(self, index) -> np.ndarray:
		return self.data[0][index], self.data[0][index]

def simple_data_loaders(train_batch_size: int, test_batch_size: int, transform = None) -> Tuple[DataLoader, DataLoader]:
		if transform:
			transformations = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), transform])
		else:
			transformations = torchvision.transforms.ToTensor()
		
		train_data = _SimpleDataset(transform=transformations, train=True)
		test_data = _SimpleDataset(transform=transformations, train=False)
		train_data_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=False, pin_memory=True)
		test_data_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)
		return train_data_loader, test_data_loader
