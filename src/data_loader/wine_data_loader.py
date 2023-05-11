import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

from typing import List, Tuple

class _WineQualityDataset(Dataset):
	def __init__(self: "_WineQualityDataset", file_path: str, train: bool = True, transform = None) -> None:
		self.data = pd.read_csv(file_path, sep=';')

		train_dataset = self.data.sample(frac=0.8,random_state=200)
		test_dataset = self.data.drop(train_dataset.index)

		if train:
			self.data = train_dataset
		else:
			self.data = test_dataset
		
		self.data = torch.Tensor(np.array(self.data))
		if transform:
			self.data = transform(self.data)


	def __len__(self: "_WineQualityDataset") -> int:
		return len(self.data)

	def __getitem__(self: "_WineQualityDataset", index: List[int]) -> np.ndarray:
		if torch.is_tensor(index):
			index = index.tolist()

		sample = self.data[index]

		return sample, sample	

def wine_quality_loaders(train_batch_size: int, test_batch_size: int, transform=None, 
			 file_path: str = './data/WineQuality/winequality-white.csv') -> Tuple[DataLoader, DataLoader]:
	train_data = _WineQualityDataset(file_path=file_path, train=True, transform=transform)
	test_data = _WineQualityDataset(file_path=file_path, train=False, transform=transform)
	train_data_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, pin_memory=True)
	test_data_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)
	return train_data_loader, test_data_loader

