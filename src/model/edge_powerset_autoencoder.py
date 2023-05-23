import torch
import torch.nn as nn
from itertools import chain, combinations

from typing import List, Callable

def powerset(iterable):
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))

class EdgePowersetLayer(nn.Module):
	def __init__(
		self: "EdgePowersetLayer",
		in_features: int,
		out_features: int,
		operator: Callable[[torch.Tensor], torch.Tensor],
		device: torch.device
	) -> None:
		super().__init__()

		torch.manual_seed(0)
		self.device = device
		self.operator: Callable[[torch.Tensor], torch.Tensor] = operator
		
		self.weights = nn.parameter.Parameter(torch.randn(2 ** in_features, device=self.device), requires_grad=True)
		self.in_features: int = in_features
		self.out_features: int = out_features

	def forward(self: "EdgePowersetLayer", x: torch.Tensor) -> torch.Tensor:
		result = torch.zeros(x.shape[0], self.out_features).to(self.device)
		for i, batch in enumerate(x):
			for u in range(self.out_features):
				input_edge_combinations = powerset(batch)
				
				num_edges = 2 ** self.in_features
				prob = nn.functional.softmax(self.weights, dim=-1)
				values = torch.zeros(num_edges).to(self.device)
				for edge_index in range(num_edges):
					if len(input_edge_combinations[edge_index]) != 0:
						values[edge_index] = (prob[edge_index] * self.operator(torch.Tensor(input_edge_combinations[edge_index])))
				result[i][u] = torch.sum(values)
		return result
		
		

class EdgePowersetAutoencoder(nn.Module):
	def __init__(
		self: "EdgePowersetAutoencoder", 
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
			layers += [EdgePowersetLayer(layer_sizes[i], layer_sizes[i + 1], self.operators[i], device)]

		self.net = nn.Sequential(*layers).to(self.device)
	
	def forward(self: "EdgePowersetAutoencoder", x: torch.Tensor):
		x = x.to(self.device)
		return self.net(x)

