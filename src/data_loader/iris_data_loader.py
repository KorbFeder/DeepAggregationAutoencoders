import pandas as pd

from data_loader.csv_data_loader import csv_loaders

def iris_loaders(train_batch_size: int, test_batch_size: int, transform=None, 
			 file_path: str = './data/Iris.csv'):
	data = pd.read_csv(file_path) 
	data['Species'] = pd.factorize(data['Species'])[0] + 1
	data = data.drop('Id', axis=1)
	data = data.drop('Species', axis=1)
	return csv_loaders(train_batch_size, test_batch_size, data, transform)

