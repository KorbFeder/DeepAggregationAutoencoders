from torch.utils.data import DataLoader

from typing import Tuple

from data_loader.mnist_data_loader import mnist_loaders
from data_loader.wine_data_loader import wine_quality_loaders
from data_loader.simple_data_loader import simple_data_loaders
from data_loader.single_data_loader import single_data_loaders
from data_loader.iris_data_loader import iris_loaders
from data_loader.csv_data_loader import csv_loaders
from globals.data_set_name import DatasetName

def get_dataloader(dataset_name: DatasetName, train_batch_size = 32, test_batch_size = 1, transform = None) -> Tuple[DataLoader, DataLoader]:
	if dataset_name == DatasetName.mnist.value:
		return mnist_loaders(train_batch_size, test_batch_size)
	elif dataset_name == DatasetName.wine.value:
		return wine_quality_loaders(train_batch_size=train_batch_size, test_batch_size=test_batch_size, transform=transform)
	elif dataset_name == DatasetName.simple.value:
		return simple_data_loaders(train_batch_size=train_batch_size, test_batch_size=test_batch_size, transform=transform)
	elif dataset_name == DatasetName.single.value:
		return single_data_loaders(train_batch_size=train_batch_size, test_batch_size=test_batch_size, transform=transform)
	elif dataset_name == DatasetName.autos.value:
		return csv_loaders(train_batch_size=train_batch_size, test_batch_size=test_batch_size, transform=transform, file_path='./data/Automobile_data.csv')
	elif dataset_name == DatasetName.iris.value:
		return iris_loaders(train_batch_size=train_batch_size, test_batch_size=test_batch_size, transform=transform, file_path='./data/Iris.csv')
	elif dataset_name == DatasetName.flag.value:
		return csv_loaders(train_batch_size=train_batch_size, test_batch_size=test_batch_size, transform=transform, file_path='./data/flag.csv')
	elif dataset_name == DatasetName.sonar.value:
		return csv_loaders(train_batch_size=train_batch_size, test_batch_size=test_batch_size, transform=transform, file_path='./data/sonar-all-data.csv')
	else:
		assert 'wrong name for dataset'
