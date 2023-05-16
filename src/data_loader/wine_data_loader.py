import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

from typing import List, Tuple

scaler = MinMaxScaler()

class _WineQualityDataset(Dataset):
	def __init__(self: "_WineQualityDataset", file_path: str, train: bool = True, transform = None) -> None:
		self.data = pd.read_csv(file_path, sep=';')
		scaler.fit(self.data)

		train_dataset = self.data.sample(frac=0.8,random_state=200)
		test_dataset = self.data.drop(train_dataset.index)

		train_dataset = scaler.transform(train_dataset)
		test_dataset = scaler.transform(test_dataset)

		if train:
			self.data = train_dataset
		else:
			self.data = test_dataset
		
		self.data = torch.Tensor(np.array(self.data))
		if transform:
			self.data = transform(self.data)

		#self.data = pd.read_csv(file_path, sep=';')
		#scaler.fit(self.data)
		#dataset_length = self.data.shape[0]

		#self.data = torch.Tensor(np.array(self.data))
		#self.data = self.data[torch.randperm(dataset_length)]

		#datasets = torch.split(self.data, int(dataset_length * 0.8))
		#train_dataset = datasets[0]
		#test_dataset = datasets[1]
		
		#train_dataset = scaler.transform(train_dataset)
		#test_dataset = scaler.transform(test_dataset)
		
		#if transform:
		#	self.data = torch.stack([t.fuzzification(self.data) for t in transform])

		#if train:
		#	self.data = train_dataset
		#else:
		#	self.data = test_dataset
		
		#self.data = torch.Tensor(np.array(self.data))


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

