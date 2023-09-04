import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from pandas.api.types import is_numeric_dtype

from typing import List, Tuple, Optional, Dict

scaler = MinMaxScaler()

class _Dataset(Dataset):
	def __init__(self: "_Dataset", data: pd.DataFrame, train: bool = True, transform = None) -> None:
		self.data = data
		self.data = self.data.dropna()
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

def csv_loaders(train_batch_size: int, test_batch_size: int, file_path: str, transform=None, 
		header = 'infer', conv_categorical=True, dataframe: Optional[pd.DataFrame] = None) -> Tuple[DataLoader, DataLoader]:
	if isinstance(dataframe, pd.DataFrame):	
		data = dataframe
	else:
		data = pd.read_csv(file_path, header=header) 

	if conv_categorical:
		data = convert_categorical(data)

	train_data = _Dataset(data=data, train=True, transform=transform)
	test_data = _Dataset(data=data, train=False, transform=transform)
	train_data_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=False, pin_memory=True)
	test_data_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)
	return train_data_loader, test_data_loader

def convert_categorical(data: pd.DataFrame):
	""" This function converts categorical values into multiple columns with each of the 
	unique values meaning. For example if a categorical columns values would be 'M' or 'F' or 'D', 
	than it would be converted to three columns, one for each categorical value and it would have the value 
	0 or 1 depending if the value of the corresponding categorical value is the same of the original column.
	0	1	2	3
	-------------
	M	1	0	0
	F	0	1	0
	D	0	0	1
	M	1	0	0

	Args:
		data (pd.DataFrame): Dataframe that will get changed

	Returns:
		The same dataframe after it got changed
	"""
	categorical_columns = data.select_dtypes(exclude=['int64','float64'])
	n_cols = len(data.columns)
	for col_name in categorical_columns.columns:
		for unique_value in data[col_name].unique():
			data[n_cols] = np.where(data[col_name] == unique_value, 1.0, 0.0)
			n_cols += 1
		del data[col_name]
	return data
