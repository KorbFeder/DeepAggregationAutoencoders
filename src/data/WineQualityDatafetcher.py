import torch
import torchvision
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

from data.Datafetcher import Datafetcher

from typing import List

class _WineQualityDataset(Dataset):
	def __init__(self: "_WineQualityDataset", file_path: str, train: bool = True, transform = None) -> None:
		self.data = pd.read_csv(file_path, sep=';')
		self.transform = transform

		train_size = int(0.8 * len(self.data))
		test_size = len(self.data) - train_size
		train_dataset, test_dataset = random_split(self.data, [train_size, test_size])

		if train:
			self.data = train_dataset
		else:
			self.data = test_dataset


	def __len__(self: "_WineQualityDataset") -> int:
		return len(self.data)

	def __getitem__(self: "_WineQualityDataset", index: List[int]) -> np.ndarray[float]:
		if torch.is_tensor(index):
			index = index.tolist()

		sample = np.array(self.data.iloc[index])
		if self.transform:
			sample = self.transform(sample)

		return sample		

class WineQualityDatafetcher(Datafetcher):
	def __init__(self: "WineQualityDatafetcher", file_path: str, transform = None) -> None:
		if transform:
			transformations = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), transform])
		else:
			transformations = torchvision.transforms.ToTensor()

		self.train_data = _WineQualityDataset(file_path=file_path, train=True, transform=transformations)
		self.test_data = _WineQualityDataset(file_path=file_path, train=False, transform=transformations)
	
	def get_train_dataset(self: "WineQualityDatafetcher") -> _WineQualityDataset:
		return self.train_data

	def get_test_dataset(self: "WineQualityDatafetcher") -> _WineQualityDataset:
		return self.test_data

	def get_train_dataloader(self: "WineQualityDatafetcher", batch_size: int = 32) -> DataLoader:
		return DataLoader(self.train_data, batch_size=batch_size, shuffle=True, pin_memory=True)

	def get_test_dataloader(self: "WineQualityDatafetcher", batch_size: int = 32) -> DataLoader:
		return DataLoader(self.test_data, batch_size=batch_size, shuffle=False)


