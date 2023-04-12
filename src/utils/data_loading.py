import torchvision
from typing import Tuple
from torch.utils.data import DataLoader

from fuzzy_logic.Fuzzyfication import Fuzzyification
from fuzzy_logic.Membership import Membership

def load_mnist_data(batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
	transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
	train_dataset = torchvision.datasets.MNIST(root="./", train=True, transform=transform, download=True)
	test_dataset = torchvision.datasets.MNIST(root="./", train=False, transform=transform, download=True)

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
	
	return train_loader, test_loader

