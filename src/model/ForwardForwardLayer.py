import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
from utils.plotting import plot_training_loss

from typing import List

#def MNIST_loaders(train_batch_size=50000, test_batch_size=10000):
#
#    transform = Compose([
#        ToTensor(),
#        Normalize((0.1307,), (0.3081,)),
#        Lambda(lambda x: torch.flatten(x))])
#
#    train_loader = DataLoader(
#        MNIST('./datasets/', train=True,
#              download=True,
#              transform=transform),
#        batch_size=train_batch_size, shuffle=False)
#
#    test_loader = DataLoader(
#        MNIST('./datasets/', train=False,
#              download=True,
#              transform=transform),
#        batch_size=test_batch_size, shuffle=False)
#
#    return train_loader, test_loader

def MNIST_loaders(train_batch_size=50000, test_batch_size=10000):
	transform = Compose([
		ToTensor(),
		Normalize((0.1307,), (0.3081,)),
		Lambda(lambda x: torch.flatten(x))])

	pos_mnist = MNIST('./datasets/', train=True,
				download=True,
				transform=transform)

	neg_mnist = MNIST('./datasets/', train=True,
				download=True,
				transform=transform)
	
	pos_mnist_test = MNIST('./datasets/', train=False,
				download=True,
				transform=transform)
	
	neg_mnist_test = MNIST('./datasets/', train=False,
				download=True,
				transform=transform)
	

	pos_idx = (pos_mnist.targets == 0) | (pos_mnist.targets == 1) | (pos_mnist.targets == 2)| (pos_mnist.targets == 3) | (pos_mnist.targets == 4)
	neg_idx = (neg_mnist.targets == 5) | (neg_mnist.targets == 6) | (neg_mnist.targets == 7)| (neg_mnist.targets == 8) | (neg_mnist.targets == 9)
	pos_idx_test = (pos_mnist_test.targets == 0) | (pos_mnist_test.targets == 1) | (pos_mnist_test.targets == 2)| (pos_mnist_test.targets == 3) | (pos_mnist_test.targets == 4)
	neg_idx_test = (neg_mnist_test.targets == 5) | (neg_mnist_test.targets == 6) | (neg_mnist_test.targets == 7)| (neg_mnist_test.targets == 8) | (neg_mnist_test.targets == 9)
	
	pos_mnist.targets = pos_mnist.targets[pos_idx]
	neg_mnist.targets = neg_mnist.targets[neg_idx]
	pos_mnist_test.targets = pos_mnist_test.targets[pos_idx_test]
	neg_mnist_test.targets = neg_mnist_test.targets[neg_idx_test]
	
	pos_mnist.data = pos_mnist.data[pos_idx]
	neg_mnist.data = neg_mnist.data[neg_idx]
	pos_mnist_test.data = pos_mnist_test.data[pos_idx_test]
	neg_mnist_test.data = neg_mnist_test.data[neg_idx_test]

	pos_train_loader = DataLoader(pos_mnist, batch_size=train_batch_size, shuffle=False)
	neg_train_loader = DataLoader(neg_mnist, batch_size=train_batch_size, shuffle=False)
	pos_test_loader = DataLoader(pos_mnist_test, batch_size=test_batch_size, shuffle=False)
	neg_test_loader = DataLoader(neg_mnist_test, batch_size=test_batch_size, shuffle=False)

	return pos_train_loader, neg_train_loader, pos_test_loader, neg_test_loader

#class ForwardForwardNet(nn.Module):
#	def __init__(
#		self: "ForwardForwardNet",
#		in_features: int, 
#		hidden_sizes: List[int], 
#		activation: nn.Module = nn.ReLU,
#		output_activation: nn.Module = nn.Sigmoid,
#	) -> None:
#		super().__init__()
#		self.activation = activation
#		self.out_activation = output_activation
#
#		layer_sizes = [in_features, *hidden_sizes, in_features]
#		self.layers = []
#
#		for i in range(len(layer_sizes)-1):
#			self.layers += [nn.Linear(layer_sizes[i], layer_sizes[i + 1])]
#
#	def forward(self: "ForwardForwardNet", x: torch.Tensor):	
#		for layer in self.layers:
#			x = self.activation(layer(x))
#
#	def train(self: "ForwardForwardNet", pos: torch.Tensor, neg: torch.Tensor):
#		for layer in self.layers:
#			out = self.activation(layer(pos))
			





class Net(torch.nn.Module):
	def __init__(self, in_features: int, hidden_dims: List[int]):
		super().__init__()
		self.layers = [ForwardForwardLinearLayer(in_features, hidden_dims[0]).cuda()]

		for d in range(len(hidden_dims) - 1):
			self.layers += [ForwardForwardLinearLayer(hidden_dims[d],hidden_dims[d + 1]).cuda()]

		self.output_layer = OutputLayer(hidden_dims[-1], in_features).cuda()

	def forward(self, x):
		for layer in self.layers:
			x = layer(x)
		#values = torch.cat(sigm_layer_inputs)
		return self.output_layer(x)

	def train(self, x_pos, x_neg):
		h_pos, h_neg = x_pos, x_neg
		for i, layer in enumerate(self.layers):
			print('training layer', i, '...')
			h_pos, h_neg = layer.train(h_pos, h_neg)

		print('training output layer ...')
		self.output_layer.train(h_pos, x_pos)

class OutputLayer(nn.Linear):
	def __init__(
		self: "OutputLayer", 
		in_features: int, 
		out_features: int, 
		activation: nn.Module = nn.Sigmoid(),
		bias: bool = True, 
		device=None, 
		dtype=None, 
		epochs: int = 1000
	) -> None:
		super().__init__(in_features, out_features, bias, device, dtype)
		self.epochs = epochs
		self.activation = activation
		self.optim = Adam(self.parameters(), lr=0.03)
		self.loss_fn = nn.MSELoss()

	def forward(self: "OutputLayer", x: torch.Tensor):
		return self.activation(torch.mm(x, self.weight.T) + self.bias.unsqueeze(0))	

	def train(self: "OutputLayer", pos: torch.Tensor, y: torch.Tensor):
		losses = []
		for i in tqdm(range(self.epochs)):
			prediction = self(pos)
			loss = self.loss_fn(prediction, y)

			self.optim.zero_grad()
			loss.backward()
			self.optim.step()
			with torch.no_grad():
				losses.append(loss.cpu())
		
		#plot_training_loss(losses)


class ForwardForwardLinearLayer(nn.Linear):
	def __init__(
		self: "ForwardForwardLinearLayer", 
		in_features: int, 
		out_features: int, 
		activation: nn.Module = nn.ReLU(),
		epochs: int = 500,
		threshold: float = 2.0,
		bias: bool = True,
		device = None,
		dtype = None
	) -> None:
		super().__init__(in_features, out_features, bias, device, dtype)
		self.epochs = epochs
		self.activation = activation
		self.threshold = threshold
		self.optim = Adam(self.parameters(), lr=0.03)

	def forward(self: "ForwardForwardLinearLayer", x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
		#x_direction = x / (x.norm(2, 1, keepdim=True), + 1e-4)
		#return self.activation(torch.mm(x_direction, self.weight.T) + self.bias.unsqueeze(0))
		mult = torch.mm(x, self.weight.T)
		multi_with_bias = mult + self.bias.unsqueeze(0)
		x = self.activation(multi_with_bias)
		if normalize:
			norm = (x.norm(2, 1, keepdim=True) + 1e-4)
			x = x / norm
		return x


	def train(self: "ForwardForwardLinearLayer", x_pos: torch.Tensor, x_neg: torch.Tensor):
		for i in tqdm(range(self.epochs)):
			g_pos = self.forward(x_pos).pow(2).mean(1)
			g_neg = self.forward(x_neg).pow(2).mean(1)

			loss = torch.log(1+torch.exp(torch.cat([
				-g_pos + self.threshold,
				g_neg - self.threshold
			]))).mean()

			self.optim.zero_grad()
			loss.backward()
			self.optim.step()
		return self.forward(x_pos).detach(), self.forward(x_neg).detach()

def visualize_sample(data, name='', idx=0):
	with torch.no_grad():
		reshaped = data[idx].cpu().reshape(28, 28)
		plt.figure(figsize = (4, 4))
		plt.title(name)
		plt.imshow(reshaped, cmap="gray")
		plt.show()
 

def overlay_y_on_x(x, y):
    """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
    """
    x_ = x.clone()
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_

def test():
	torch.manual_seed(1234)
	pos_train_loader, neg_train_loader, pos_test_loader, neg_test_loader = MNIST_loaders()
	#train_loader, test_loader = MNIST_loaders()

	net = Net(784, [256, 64, 256])
	x, _ = next(iter(pos_train_loader))
	x_pos = x.cuda()
	#x_pos = overlay_y_on_x(x, y)
	#rnd = torch.randperm(x.size(0))
	_x, _ = next(iter(neg_train_loader))
	x_neg = _x.cuda()
	#x_neg = overlay_y_on_x(x, y[rnd])
	#x, y = next(iter(train_loader))
	#x, y = x.cuda(), y.cuda()
	#x_pos = overlay_y_on_x(x, y)
	#rnd = torch.randperm(x.size(0))
	#x_neg = overlay_y_on_x(x, y[rnd])
  

	#for data, name in zip([x, x_pos, x_neg], ['orig', 'pos', 'neg']):
	#	visualize_sample(data, name)

	net.train(x_pos, x_neg)

	x_te, y_te = next(iter(pos_test_loader))
	#x_te, y_te = next(iter(test_loader))
	x_te, y_te = x_te.cuda(), y_te.cuda()

	data = net(x_te)
	visualize_sample(x_te, 'original')
	visualize_sample(data, 'result')