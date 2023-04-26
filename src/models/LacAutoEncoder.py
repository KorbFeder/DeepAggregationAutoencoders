import torch
import torch.nn as nn
import torch.functional as F
from torch.optim import Adam, SGD
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from typing import List, Tuple

def plot_training_loss(losses: List[float], name: str = 'default') -> None:
	plt.plot(losses)
	plt.title = 'Training Loss'
	plt.xlabel = 'Epochs'
	plt.ylabel = 'Loss'
	plt.savefig(f'./images/training_loss_curve-{name}.png')

def MNIST_loaders(train_batch_size=1, test_batch_size=1):
    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))])

    train_loader = DataLoader(
        MNIST('./datasets/', train=True,
              download=True,
              transform=transform),
        batch_size=train_batch_size, shuffle=False)

    test_loader = DataLoader(
        MNIST('./datasets/', train=False,
              download=True,
              transform=transform),
        batch_size=test_batch_size, shuffle=False)
    return train_loader, test_loader

def visualize_sample(data, name='', idx=0):
    reshaped = data[idx].cpu().reshape(28, 28)
    plt.figure(figsize = (4, 4))
    plt.title = name
    plt.imshow(reshaped, cmap="gray")
    plt.show()
    

class LacAutoEncoder(nn.Module):
	def __init__(
		self: "LacAutoEncoder",
		in_features: int,
		hidden_sizes: List[int],
		activation: nn.Module = nn.ReLU(),
		out_activation: nn.Module = nn.Sigmoid()
	) -> None:
		super().__init__()

		layer_sizes = [in_features, *hidden_sizes, in_features]
		self.out_activation = out_activation
		self.activation = activation
		self.layers: List[nn.Linear] = []
		self.num_hidden_neurons = sum(hidden_sizes)

		for i in range(len(layer_sizes)-1):
			self.layers += [nn.Linear(layer_sizes[i], layer_sizes[i + 1])]

		self.hidden_optim = Adam(sum([list(layer.parameters()) for layer in self.layers[:-1]], []))
		self.output_optim = Adam(self.layers[-1].parameters())

	def forward(self: "LacAutoEncoder", x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		hidden_activations = torch.Tensor(self.num_hidden_neurons)
		i = 0

		# forward pass through the network, but save the activation of every neuron
		A = x
		for layer in self.layers[:-1]: 
			WA = layer(A.detach())
			S = WA / WA.norm(2, 1, keepdim=True) + 1e-4
			A = self.activation(S)
			new_i = S.shape[1] + i
			hidden_activations[i:new_i] = S
			i = new_i

		# the last layers output
		output = self.activation(self.layers[-1](A.detach()))

		return hidden_activations, output

lae = LacAutoEncoder(784, [256, 64, 256])

train_loader, test_loader = MNIST_loaders()
hidden_losses = []
output_losses = []

for x in tqdm(train_loader):
	x = x[0]

	target_hidden_activities, target_output = lae(x)
	prediction_hidden_activities, prediction_output = lae(target_output)

	hidden_loss = torch.sum((target_hidden_activities.detach() - prediction_hidden_activities)**2)
	output_loss = torch.sum((target_output - x)**2)

	# hidden layer update
	lae.hidden_optim.zero_grad()
	hidden_loss.backward(retain_graph=True)
	lae.hidden_optim.step()

	# output layer update
	lae.output_optim.zero_grad()
	output_loss.backward()
	lae.output_optim.step()

	with torch.no_grad():
		hidden_losses.append(hidden_loss)
		output_losses.append(output_loss)

with torch.no_grad():
	plot_training_loss(hidden_losses, "hidden loss")
	plot_training_loss(output_losses, "output loss")

	x = next(iter(test_loader))[0]
	x_new = lae(x)

	visualize_sample(x)
	visualize_sample(x_new)

