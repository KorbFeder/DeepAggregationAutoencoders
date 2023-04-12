import torch
from torch import nn

from typing import List, Callable

def power(x: torch.Tensor, exponent: torch.Tensor, fuzzy_operator: Callable[[torch.Tensor], torch.Tensor]):
	n_next_layer_neurons = exponent.shape[0]
	n_samples = x.shape[0]
	result = torch.Tensor(n_samples, n_next_layer_neurons)
	for u in range(n_samples):
		for i in range(n_next_layer_neurons):
			result[u][i] = fuzzy_operator(torch.pow(x[u], exponent[i]))
	return result


class FuzzyHedgeLayer(nn.Module):
	def __init__(
		self: "FuzzyHedgeLayer", 
		in_features: int, 
		out_features: int, 
		fuzzy_operator: Callable[[torch.Tensor], torch.Tensor]
	) -> None:
		super().__init__()
		self.fuzzy_operator = fuzzy_operator
		self.exponent = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=True)
		nn.init.uniform_(self.exponent, a=0, b=100)

	def forward(self: "FuzzyHedgeLayer", x: torch.Tensor) -> torch.Tensor:
		return power(x, self.exponent, self.fuzzy_operator)

class DeepAggregateAutoEncoder(nn.Module):
	def __init__(self: "DeepAggregateAutoEncoder",
		in_features: int, 
		hidden_sizes: List[int], 
		fuzzy_operators_per_layer: List[Callable[[torch.Tensor], torch.Tensor]],
		activation: nn.Module = nn.ReLU, 
		output_activation: nn.Module = nn.Identity
	) -> None:
		super().__init__()

		layer_sizes = [in_features, *hidden_sizes, in_features]
		layers = []

		for i in range(len(layer_sizes)-1):
			activation_function = activation if i < len(layer_sizes) - 2 else output_activation
			layers += [FuzzyHedgeLayer(layer_sizes[i], layer_sizes[i + 1], fuzzy_operators_per_layer[i]), activation_function()]

		self.net = nn.Sequential(*layers)

	def forward(self: "DeepAggregateAutoEncoder", x: torch.Tensor):
		return self.net(x)
