import torch
import torch.nn as nn
from enum import Enum
from functools import partial
from fuzzy_logic.fuzzy_operators import fuzzy_alg, fuzzy_coalg

from typing import List, Callable, Optional

class EdgeType(Enum):
	#no_edge = partial(lambda x, op = torch.min: torch.ones(x.shape) if op == torch.min else torch.zeros(x.shape))
	no_edge = partial(lambda x, op = fuzzy_alg: torch.zeros(x.shape) if op == fuzzy_alg else torch.zeros(x.shape))
	normal_edge = partial(lambda x, _: x)
	#very = partial(lambda x, _: torch.square(x))
	#somewhat = partial(lambda x, _: torch.sqrt(x + 1.e-8))
	#Not = partial(lambda x, _: 1 - x)

class DiffEdgeLayer(nn.Module):
	def __init__(
		self: "DiffEdgeLayer",
		in_features: int,
		out_features: int,
		operator: Callable[[torch.Tensor], torch.Tensor],
		device: torch.device,
		seed: Optional[int] = None
	) -> None:
		super().__init__()

		if seed != None:
			torch.manual_seed(seed)
		
		self.edge_types: List[Callable[[torch.Tensor], torch.Tensor]] = [etype.value for etype in EdgeType]

		self.device = device
		self.operator: Callable[[torch.Tensor], torch.Tensor] = operator
		
		self.prob_weights = nn.parameter.Parameter(torch.randn(out_features, in_features, len(EdgeType), device=self.device), requires_grad=True)
		self.in_features: int = in_features
		self.out_features: int = out_features

	def forward(self: "DiffEdgeLayer", x: torch.Tensor, is_train: bool = False) -> torch.Tensor:
		if is_train:
			prob = nn.functional.softmax(self.prob_weights, dim=-1)
		else:
			prob = torch.nn.functional.one_hot(self.prob_weights.argmax(-1), len(EdgeType)).to(torch.float32)

		edge_type_values = torch.zeros(len(EdgeType), x.shape[0], self.out_features, self.in_features).to(self.device)

		# calculate the probabilty times the value of the edge for each edge type
		for i, edge_type in enumerate(self.edge_types):
			edge_prob = prob[..., i]
			edge_value = edge_type(x, self.operator).to(self.device)

			multiplication_result = edge_value[:, None] *  edge_prob
			edge_type_values[i] = multiplication_result

		expected_value = torch.sum(edge_type_values, dim=0)
		return self.operator(expected_value, dim=-1)
			
class DiffEdgeAutoencoder(nn.Module):
	def __init__(
		self: "DiffEdgeAutoencoder", 
		in_features: int,
		hidden_sizes: List[int],
		device: torch.device,
		operators: List[Callable[[torch.Tensor], torch.Tensor]] = [fuzzy_alg, fuzzy_coalg],
		seed: Optional[int] = None
	) -> None:
		super().__init__()
		self.device = device
		self.in_features = in_features
		layer_sizes = [in_features, *hidden_sizes, in_features]
		self.operators = (operators * int((len(layer_sizes) / len(operators)) + 1))[:len(layer_sizes)]
		#self.operators = [torch.min if i % 2 ==  0 else torch.max for i in range(len(layer_sizes) - 1)]
		self.layers = []

		for i in range(len(layer_sizes)-1):
			self.layers += [DiffEdgeLayer(layer_sizes[i], layer_sizes[i + 1], self.operators[i], device, seed)]

		self.net = nn.Sequential(*self.layers).to(self.device)
	
	def forward(self: "DiffEdgeAutoencoder", x: torch.Tensor, is_train: bool = False) -> torch.Tensor:
		x = x.to(self.device)
		#is_train = True
		for layer in self.layers:
			x = layer(x, is_train)
			x = x.to(self.device)
		return x
