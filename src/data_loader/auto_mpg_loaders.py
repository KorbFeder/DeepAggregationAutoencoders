import pandas as pd

from data_loader.csv_data_loader import csv_loaders

def auto_mpg_loaders(train_batch_size: int, test_batch_size: int, transform=None, 
			 file_path: str = './data/auto_mpg/auto-mpg.data'):
	data = pd.read_csv(file_path, na_values='?')
	data = data.drop(data.columns[-1], axis=1)

	return csv_loaders(train_batch_size=train_batch_size, test_batch_size=test_batch_size, transform=transform, 
		     file_path=file_path, dataframe=data)

