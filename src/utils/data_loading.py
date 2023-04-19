import torchvision
from typing import Tuple
from torch.utils.data import DataLoader

def get_mnist_dataloaders(batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
	transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
	train_dataset = torchvision.datasets.MNIST(root="./data/", train=True, transform=transform, download=True)
	test_dataset = torchvision.datasets.MNIST(root="./data/", train=False, transform=transform, download=True)

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
	
	return train_loader, test_loader