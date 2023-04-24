from torch.utils.data import Dataset, DataLoader
from abc import ABC, abstractclassmethod

class Datafetcher(ABC):
	@abstractclassmethod
	def get_train_dataset(self: "Datafetcher") -> Dataset:
		raise NotImplementedError

	@abstractclassmethod
	def get_test_dataset(self: "Datafetcher") -> Dataset:
		raise NotImplementedError

	@abstractclassmethod
	def get_train_dataloader(self: "Datafetcher", batch_size: int) -> DataLoader:
		raise NotImplementedError

	@abstractclassmethod
	def get_test_dataloader(self: "Datafetcher", batch_size: int) -> DataLoader:
		raise NotImplementedError