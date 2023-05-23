import torch
import torch.nn as nn
import math
import random

from typing import List, Callable

edge_type = [0, 1]

class EdgeLayer(nn.Module):
	def __init__(
		self: "EdgeLayer",
		in_features: int,
		out_features: int,
		operator: Callable[[torch.Tensor], torch.Tensor],
		device: torch.device
	) -> None:
		super().__init__()

		torch.manual_seed(0)
		self.device = device
		self.operator: Callable[[torch.Tensor], torch.Tensor] = operator
		self.edges = nn.parameter.Parameter(torch.randn(out_features, in_features, len(edge_type), device=self.device), requires_grad=True)
		self.in_features: int = in_features
		self.out_features: int = out_features

	# if there is a node without any connection, give it a connection
	def zero_connection_neuron(self: "EdgeLayer", mask: torch.Tensor) -> torch.Tensor:
		connection_count = torch.count_nonzero(mask, dim=-1)
		no_conn_indices= (connection_count == 0).nonzero()
		for index in no_conn_indices:
			connection = random.randint(0, mask.shape[1] - 1)
			mask[index.item()][connection] += 1
		return mask

	def forward(self: "EdgeLayer", x: torch.Tensor) -> torch.Tensor:
		print(self.edges)
		prob = nn.functional.softmax(self.edges, dim=-1)
		prob_of_edge = prob[..., 1]	

		mask = torch.bernoulli(prob_of_edge)
		mask = self.zero_connection_neuron(mask)
		
		if self.operator == torch.max:
			mask *= -1

		result = torch.zeros(x.shape[0], mask.shape[0]).to(self.device)
		for i, batch in enumerate(x):
			operator_inputs = batch.add(mask)
			result[i] = self.operator(operator_inputs, dim=-1).values

		return result

		#edge_state = torch.nn.functional.one_hot(self.edges.argmax(-1), len(edge_type)).to(torch.float32)
		#curr_edges = list([list(i for i, edge in enumerate(neuron) if edge != 0) for neuron in edge_state[..., 1]])
		##connected_x = torch.cat([x[..., edges] for edges in curr_edges])
		#for edge in curr_edges:
		#	connected_x = x[..., edge]
		#self.operator(connected_x, dim=-1)
		
class EdgeAutoencoder(nn.Module):
	def __init__(
		self: "EdgeAutoencoder", 
		in_features: int,
		hidden_sizes: List[int],
		device: torch.device
	) -> None:
		super().__init__()
		self.device = device
		self.in_features = in_features
		layer_sizes = [in_features, *hidden_sizes, in_features]
		self.operators = [torch.min if i % 2 ==  0 else torch.max for i in range(len(layer_sizes) - 1)]
		layers = []

		for i in range(len(layer_sizes)-1):
			layers += [EdgeLayer(layer_sizes[i], layer_sizes[i + 1], self.operators[i], device)]

		self.net = nn.Sequential(*layers).to(self.device)
	
	def forward(self: "EdgeAutoencoder", x: torch.Tensor):
		x = x.to(self.device)
		return self.net(x)


def test():
	features = torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.5])
	weights = nn.parameter.Parameter(torch.rand(5))
	



test()