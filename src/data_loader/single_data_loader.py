import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader

from typing import Tuple

class _SingleDataset(Dataset):
	def __init__(self: "_SingleDataset", transform=None, train: bool = False, length: int = 4) -> None:
		self.data = np.array([
			#[(i / 10) % 1 for i in range(length)]
			#[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0, 0.1, 0.2],
			[0.1, 0.2, 0.7, 0.8],
		], dtype=np.float32)

		self.train_data = np.array(self.data)

		self.data = np.array([val for val in self.data for _ in (range(1000))])

		if not train:
			self.data = self.train_data

		if transform:
			self.data = transform(self.data)
	
	def __len__(self: "_SingleDataset") -> int:
		return len(self.data[0])

	def __getitem__(self: "_SingleDataset", index) -> np.ndarray:
		return self.data[0][index], self.data[0][index]

def single_data_loaders(train_batch_size: int, test_batch_size: int, transform = None) -> Tuple[DataLoader, DataLoader]:
		if transform:
			transformations = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), transform])
		else:
			transformations = torchvision.transforms.ToTensor()
		
		train_data = _SingleDataset(transform=transformations, train=True)
		test_data = _SingleDataset(transform=transformations, train=False)
		train_data_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=False, pin_memory=True)
		test_data_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)
		return train_data_loader, test_data_loader
