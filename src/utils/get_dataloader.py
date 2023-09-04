from torch.utils.data import DataLoader

from typing import Tuple

from data_loader.mnist_data_loader import mnist_loaders
from data_loader.wine_data_loader import wine_quality_loaders
from data_loader.simple_data_loader import simple_data_loaders
from data_loader.single_data_loader import single_data_loaders
from data_loader.iris_data_loader import iris_loaders
from data_loader.glass_loader import glass_loaders
from data_loader.csv_data_loader import csv_loaders
from data_loader.auto_mpg_loaders import auto_mpg_loaders
from data_loader.horton_loader import horton_loaders
from data_loader.energy_loader import energy_loaders
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
	elif dataset_name == DatasetName.glass.value:
		return glass_loaders(train_batch_size=train_batch_size, test_batch_size=test_batch_size, transform=transform)
	elif dataset_name == DatasetName.bupa.value:
		return csv_loaders(train_batch_size=train_batch_size, test_batch_size=test_batch_size, transform=transform, 
		     file_path='./data/bupa/bupa.data', header=None)
	elif dataset_name == DatasetName.auto_mpg.value:
		return auto_mpg_loaders(train_batch_size=train_batch_size, test_batch_size=test_batch_size, transform=transform, 
		     file_path='./data/auto_mpg/auto-mpg.csv')
	elif dataset_name == DatasetName.abalone.value:
		return csv_loaders(train_batch_size=train_batch_size, test_batch_size=test_batch_size, transform=transform, 
		     file_path='./data/abalone/abalone.data', header=None)
	elif dataset_name == DatasetName.horton.value:
		return horton_loaders(train_batch_size=train_batch_size, test_batch_size=test_batch_size, transform=transform)
	elif dataset_name == DatasetName.energy.value:
		return energy_loaders(train_batch_size=train_batch_size, test_batch_size=test_batch_size, transform=transform)
	
	else:
		assert 'wrong name for dataset'
