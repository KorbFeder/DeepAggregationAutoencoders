import torch
import torch.nn as nn

from typing import List

class OwaLayer(nn.Module):
	def __init__(
		self: "OwaLayer",
		in_features: int,
		out_features: int,
		device: torch.device,
	) -> None:
		super().__init__()
		self.device = device
		self.in_features = in_features
		self.out_features = out_features
		self.weights = nn.parameter.Parameter(torch.randn(out_features, in_features, device=self.device), requires_grad=True)

	def forward(self: "OwaLayer", x: torch.Tensor) -> torch.Tensor:
		sorted_input = torch.sort(x, dim=-1, descending=True, stable=True).values
		prob_weights = nn.functional.softmax(self.weights, dim=-1)

		#min_max_diff = (torch.max(self.weights, dim=-1).values - torch.min(self.weights, dim=-1).values)
		#weight_minus_min = (self.weights.t() - torch.min(self.weights, dim=-1).values)
		#norm_weights = (weight_minus_min / min_max_diff)

		return torch.mm(sorted_input, prob_weights.t())
		#return torch.mm(sorted_input, self.weights.t())

class OwaAutoencoder(nn.Module):
	def __init__(
		self: "OwaAutoencoder",
		in_features: int,
		hidden_sizes: List[int],
		device: torch.device,
	) -> None:
		super().__init__()
		self.device = device
		self.in_features = in_features
		layer_sizes = [in_features, *hidden_sizes, in_features]
		self.layers = []

		for i in range(len(layer_sizes)-1):
			self.layers += [OwaLayer(layer_sizes[i], layer_sizes[i + 1], device)]

		self.net = nn.Sequential(*self.layers).to(self.device)
	
	def forward(self: "OwaAutoencoder", x: torch.Tensor):
		return self.net(x)