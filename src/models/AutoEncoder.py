import torch
from torch import nn
from typing import Tuple, List

class AutoEncoder(nn.Module):
	def __init__(self: "AutoEncoder", in_features: int, hidden_sizes: List[int], activation: nn.Module = nn.ReLU) -> None:
		super().__init__()
		
		layer_sizes = [in_features, *hidden_sizes, in_features]
		out_activation = nn.ReLU
		layers = []

		for i in range(len(layer_sizes)-1):
			activation_function = activation if i < len(layer_sizes) - 2 else out_activation
			layers += [nn.Linear(layer_sizes[i], layer_sizes[i + 1]), activation_function()]

		self.net = nn.Sequential(*layers)

	def forward(self: "AutoEncoder", features: torch.Tensor) -> torch.Tensor:
		return self.net(features)

