import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

from data.Datafetcher import Datafetcher

from typing import List

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

		return sample		

class WineQualityDatafetcher(Datafetcher):
	def __init__(self: "WineQualityDatafetcher", file_path: str, transform = None) -> None:
		self.train_data = _WineQualityDataset(file_path=file_path, train=True, transform=transform)
		self.test_data = _WineQualityDataset(file_path=file_path, train=False, transform=transform)
	
	def get_train_dataset(self: "WineQualityDatafetcher") -> _WineQualityDataset:
		return self.train_data

	def get_test_dataset(self: "WineQualityDatafetcher") -> _WineQualityDataset:
		return self.test_data

	def get_train_dataloader(self: "WineQualityDatafetcher", batch_size: int = 32) -> DataLoader:
		return DataLoader(self.train_data, batch_size=batch_size, shuffle=True, pin_memory=True)

	def get_test_dataloader(self: "WineQualityDatafetcher", batch_size: int = 32) -> DataLoader:
		return DataLoader(self.test_data, batch_size=batch_size, shuffle=False)

	def num_features(self: "WineQualityDatafetcher") -> int:
		return self.train_data.data.shape[1]

