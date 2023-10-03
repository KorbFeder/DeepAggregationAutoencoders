import pandas as pd

from data_loader.csv_data_loader import csv_loaders

def household_loaders(train_batch_size: int, test_batch_size: int, transform=None, 
			 file_path: str = './data/household/household_power_consumption.csv'):
	data = pd.read_csv(file_path, sep = ';')
	data = data.drop([data.columns[0], data.columns[1]], axis=1)

	return csv_loaders(train_batch_size=train_batch_size, test_batch_size=test_batch_size, transform=transform, 
		     file_path=file_path, dataframe=data, conv_categorical=False)


