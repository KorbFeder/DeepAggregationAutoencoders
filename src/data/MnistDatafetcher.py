import torchvision
from torch.utils.data import DataLoader, Dataset

from data.Datafetcher import Datafetcher

class MnistDatafetcher(Datafetcher):
	def __init__(self: "MnistDatafetcher", file_save_path: str = "./datasets/", transform=None) -> None:
		if transform:
			transformations = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), transform])
		else:
			transformations = torchvision.transforms.ToTensor()

		self.train_dataset = torchvision.datasets.MNIST(root=file_save_path, train=True, transform=transformations, download=True)
		self.test_dataset = torchvision.datasets.MNIST(root=file_save_path, train=False, transform=transformations, download=True)

	def get_train_dataset(self: "MnistDatafetcher") -> Dataset:
		return self.train_dataset

	def get_test_dataset(self: "MnistDatafetcher") -> Dataset:
		return self.test_dataset

	def get_train_dataloader(self: "MnistDatafetcher", batch_size: int = 32) -> DataLoader:
		return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

	def get_test_dataloader(self: "MnistDatafetcher", batch_size: int = 32) -> DataLoader:
		return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
