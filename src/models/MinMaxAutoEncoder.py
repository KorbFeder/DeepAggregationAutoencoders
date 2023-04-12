import torch
import math
from torch import nn

from typing import List, Callable

class MinMaxLayer(nn.Module):
	def __init__(
		self: "MinMaxLayer", 
		in_features: int, 
		out_features: int, 
		fuzzy_operator: Callable[[torch.Tensor], torch.Tensor]
	) -> None:
		super().__init__()
		self.fuzzy_operator = fuzzy_operator
		self.weights = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=True)
		self.bias = nn.Parameter(torch.Tensor(out_features))

		nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
		fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
		bound = 1 / math.sqrt(fan_in)
		nn.init.uniform_(self.bias, -bound, bound)

	def forward(self: "MinMaxLayer", x: torch.Tensor) -> torch.Tensor:
		return self.fuzzy_operator(torch.mm(x, self.weights.t()), self.bias)

class MinMaxAutoEncoder(nn.Module):
	def __init__(self: "MinMaxAutoEncoder",
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
			layers += [MinMaxLayer(layer_sizes[i], layer_sizes[i + 1], fuzzy_operators_per_layer[i]), activation_function()]

		self.net = nn.Sequential(*layers)

	def forward(self: "MinMaxAutoEncoder", x: torch.Tensor):
		return self.net(x)

