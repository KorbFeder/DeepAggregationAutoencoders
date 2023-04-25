import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader

from typing import List


def MNIST_loaders(train_batch_size=50000, test_batch_size=10000):

    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))])

    train_loader = DataLoader(
        MNIST('./datasets/', train=True,
              download=True,
              transform=transform),
        batch_size=train_batch_size, shuffle=True)

    test_loader = DataLoader(
        MNIST('./datasets/', train=False,
              download=True,
              transform=transform),
        batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader


class Net(torch.nn.Module):
	def __init__(self, in_features: int, hidden_dims: List[int]):
		super().__init__()
		self.layers = [ForwardForwardLinearLayer(in_features, hidden_dims[0]).cuda()]
		for d in range(len(hidden_dims) - 1):
			self.layers += [ForwardForwardLinearLayer(hidden_dims[d],hidden_dims[d + 1]).cuda()]
		# the softmax layer, the actvation of every layer except the first layer 
		self.softmax_layer = ForwardForwardLinearLayer(sum(hidden_dims[1:]), in_features, activation=nn.Softmax()).cuda()

	def forward(self, x):
		softmax_layer_inputs = []
		for i, layer in enumerate(self.layers):
			x = layer(x)
			if i != 0: 
				softmax_layer_inputs.append(x)
		#values = torch.cat(sigm_layer_inputs)
		return self.softmax_layer(torch.cat(softmax_layer_inputs, dim=1), normalize = False)

	def train(self, x_pos, x_neg):
		softmax_layer_pos_inputs = []
		softmax_layer_neg_inputs = []

		h_pos, h_neg = x_pos, x_neg
		for i, layer in enumerate(self.layers):
			print('training layer', i, '...')
			h_pos, h_neg = layer.train(h_pos, h_neg)
			if i != 0: 
				softmax_layer_pos_inputs.append(h_pos)
				softmax_layer_neg_inputs.append(h_pos)

		print('training softmax layer (output layer) ...')
		h_pos = torch.cat(softmax_layer_pos_inputs, dim=1)
		h_neg = torch.cat(softmax_layer_neg_inputs, dim=1)
		h_pos, h_neg = self.softmax_layer.train(h_pos, h_neg)


class ForwardForwardLinearLayer(nn.Linear):
	def __init__(
		self: "ForwardForwardLinearLayer", 
		in_features: int, 
		out_features: int, 
		activation: nn.Module = nn.ReLU(),
		epochs: int = 1000,
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
 

def test():
	torch.manual_seed(1234)
	train_loader, test_loader = MNIST_loaders()

	net = Net(784, [256, 64, 256])
	x, _ = next(iter(train_loader))
	x_pos = x.cuda()
	#x_pos = overlay_y_on_x(x, y)
	#rnd = torch.randperm(x.size(0))
	x_neg = torch.rand(x_pos.shape).cuda()
	#x_neg = overlay_y_on_x(x, y[rnd])

	#for data, name in zip([x, x_pos, x_neg], ['orig', 'pos', 'neg']):
	#	visualize_sample(data, name)

	net.train(x_pos, x_neg)

	x_te, y_te = next(iter(test_loader))
	x_te, y_te = x_te.cuda(), y_te.cuda()

	data = net(x_te)
	visualize_sample(x_te, 'original')
	visualize_sample(data, 'result')