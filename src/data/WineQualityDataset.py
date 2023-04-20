import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, random_split

from typing import List

class WineQualityDataset(Dataset):
	def __init__(self: "WineQualityDataset", file_path: str, train: bool = True, transform = None) -> None:
		self.data = pd.read_csv(file_path, sep=';')
		self.transform = transform

		train_size = int(0.8 * len(self.data))
		test_size = len(self.data) - train_size
		train_dataset, test_dataset = random_split(self.data, [train_size, test_size])

		if train:
			self.data = train_dataset
		else:
			self.data = test_dataset


	def __len__(self: "WineQualityDataset") -> int:
		return len(self.data)

	def __getitem__(self: "WineQualityDataset", index: List[int]) -> np.ndarray[float]:
		if torch.is_tensor(index):
			index = index.tolist()

		sample = np.array(self.data.iloc[index])
		if self.transform:
			sample = self.transform(sample)

		return sample		