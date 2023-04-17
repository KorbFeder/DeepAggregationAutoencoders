import torch
import torch.nn as nn

from typing import Callable, List

class BinaryLayer(nn.Module):
	def __init__(
		self: "BinaryLayer", 
		in_features: int, 
		out_features: int, 
		fuzzy_operator: Callable[[torch.Tensor], torch.Tensor]
	) -> None:
		super().__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.fuzzy_operator = fuzzy_operator

		# initalize weights with either 0 or 1
		self.weights = nn.Parameter(torch.randint(0, 2, (self.out_features, self.in_features)), requires_grad=False)

	def forward(self: "BinaryLayer", x: torch.Tensor) -> torch.Tensor:
		return self.fuzzy_operator(self.weights[:, None] * x, dim=2).values.T

class BinaryAutoEncoder(nn.Module):
	def __init__(
		self: "BinaryAutoEncoder",
		in_features: int, 
		hidden_sizes: List[int], 
		fuzzy_operators_per_layer: List[Callable[[torch.Tensor], torch.Tensor]],
	) -> None:
		super().__init__()
	
		layer_sizes = [in_features, *hidden_sizes, in_features]
		layers = []

		for i in range(len(layer_sizes)-1):
			layers += [BinaryLayer(layer_sizes[i], layer_sizes[i + 1], fuzzy_operators_per_layer[i])]

		self.net = nn.Sequential(*layers)

	def forward(self: "BinaryAutoEncoder", x: torch.Tensor):
		return self.net(x)