import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

from typing import List, Tuple

scaler = MinMaxScaler()

class _Dataset(Dataset):
	def __init__(self: "_Dataset", data: pd.DataFrame, train: bool = True, transform = None) -> None:
		self.data = data
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

	def __len__(self: "_Dataset") -> int:
		return len(self.data)

	def __getitem__(self: "_Dataset", index: List[int]) -> np.ndarray:
		if torch.is_tensor(index):
			index = index.tolist()

		sample = self.data[index]

		return sample, sample	

def csv_loaders(train_batch_size: int, test_batch_size: int, 
			 dataframe: pd.DataFrame, transform=None) -> Tuple[DataLoader, DataLoader]:
	train_data = _Dataset(data=dataframe, train=True, transform=transform)
	test_data = _Dataset(data=dataframe, train=False, transform=transform)
	train_data_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, pin_memory=True)
	test_data_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)
	return train_data_loader, test_data_loader

