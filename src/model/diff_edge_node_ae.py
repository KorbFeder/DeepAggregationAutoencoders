import torch
import torch.nn as nn
from enum import Enum
from functools import partial
from fuzzy_logic.fuzzy_operators import T_Norm, T_Conorm
from fuzzy_logic.edge_types import EdgeType

from typing import List, Callable, Optional, Tuple, Union

NO_EDGE_OFFSET_T_NORM = 1
NO_EDGE_OFFSET_T_CONORM = -1

class TrainMode(Enum):
	no_train = 0
	train_edges = 1
	train_nodes = 2
	train_both = 3

class DiffEdgeNodeLayer(nn.Module):
	def __init__(
		self: "DiffEdgeNodeLayer",
		in_features: int,
		out_features: int,
		operators: List[Callable[[torch.Tensor], torch.Tensor]],
		edge_types: List[EdgeType],
		device: torch.device,
		seed: Optional[int] = None,
	) -> None:
		super().__init__()

		if seed != None:
			torch.manual_seed(seed)
		
		self.device = device
		self.operators: List[Callable[[torch.Tensor], torch.Tensor]] = operators
		self.edge_types = edge_types
		
		self.prob_edge_weights = nn.parameter.Parameter(
			torch.randn(out_features, in_features, len(edge_types), device=self.device), requires_grad=True)
		self.prob_node_weights = nn.parameter.Parameter(
			torch.randn(out_features, len(operators), device=self.device), requires_grad=True)

		self.in_features: int = in_features
		self.out_features: int = out_features

	def forward(self: "DiffEdgeNodeLayer", x: torch.Tensor, train_mode: TrainMode = TrainMode.no_train) -> torch.Tensor:
		edge_prob, node_prob = self.get_probabilities(train_mode)

		operator_values = torch.zeros(len(self.operators), x.shape[0], self.out_features).to(self.device)

		for u, operator in enumerate(self.operators):
			edge_type_values = torch.zeros(len(self.edge_types), x.shape[0], self.out_features, self.in_features).to(self.device)

			# calculate the probabilty times the value of the edge for each edge type
			for i, edge_type in enumerate(self.edge_types):
				curr_edge_prob = edge_prob[..., i]
				edge_value = edge_type.value(x).to(self.device)

				if edge_type == EdgeType.no_edge:
					edge_value += self.no_edge_offset(operator)

				multiplication_result = edge_value[:, None] *  curr_edge_prob
				edge_type_values[i] = multiplication_result

			expected_value = torch.sum(edge_type_values, dim=0)
			operator_result = operator.value(expected_value)
			operator_values[u] = operator_result * node_prob[..., u]
		ev_operators = torch.sum(operator_values, dim = 0)
		return ev_operators

	def get_probabilities(self: "DiffEdgeNodeLayer", train_mode: TrainMode) -> Tuple[torch.Tensor, torch.Tensor]:
		if train_mode == TrainMode.train_both:
			edge_prob = nn.functional.softmax(self.prob_edge_weights, dim=-1)
			node_prob = nn.functional.softmax(self.prob_node_weights, dim=-1)
		if train_mode == TrainMode.train_edges:
			edge_prob = nn.functional.softmax(self.prob_edge_weights, dim=-1)
			with torch.no_grad():
				node_prob = torch.nn.functional.one_hot(self.prob_node_weights.argmax(-1), len(self.operators)).to(torch.float32)
		if train_mode == TrainMode.train_nodes:
			node_prob = nn.functional.softmax(self.prob_node_weights, dim=-1)
			with torch.no_grad():
				edge_prob = torch.nn.functional.one_hot(self.prob_edge_weights.argmax(-1), len(self.edge_types)).to(torch.float32)
		if train_mode == TrainMode.no_train:
			with torch.no_grad():
				edge_prob = torch.nn.functional.one_hot(self.prob_edge_weights.argmax(-1), len(self.edge_types)).to(torch.float32)
				node_prob = torch.nn.functional.one_hot(self.prob_node_weights.argmax(-1), len(self.operators)).to(torch.float32)
		return edge_prob, node_prob
	
	def no_edge_offset(self: "DiffEdgeNodeLayer", operator: Union[T_Norm, T_Conorm]) -> int:
		if operator.value in set(item.value for item in T_Norm):
			return NO_EDGE_OFFSET_T_NORM
		if operator.value in set(item.value for item in T_Conorm):
			return NO_EDGE_OFFSET_T_CONORM

			
class DiffEdgeNodeAutoencoder(nn.Module):
	def __init__(
		self: "DiffEdgeNodeAutoencoder", 
		in_features: int,
		hidden_sizes: List[int],
		device: torch.device,
		operators: List[Union[T_Norm, T_Conorm]] = [T_Norm.min, T_Conorm.max],
		edge_types: List[EdgeType] = [EdgeType.no_edge, EdgeType.normal_edge], #, EdgeType.very, EdgeType.somewhat, EdgeType.Not],
		use_weights_after_train: bool = False,
		seed: Optional[int] = None
	) -> None:
		super().__init__()
		self.device = device
		self.in_features = in_features
		layer_sizes = [in_features, *hidden_sizes, in_features]
		self.layers = []
		self.use_weights_after_train = use_weights_after_train
		self.saved_train_mode = TrainMode.no_train
		self.edge_types = edge_types

		for i in range(len(layer_sizes)-1):
			self.layers += [DiffEdgeNodeLayer(layer_sizes[i], layer_sizes[i + 1], operators, edge_types, device, seed)]

		self.net = nn.Sequential(*self.layers).to(self.device)
	
	def forward(self: "DiffEdgeNodeAutoencoder", x: torch.Tensor,train_mode: TrainMode = TrainMode.no_train) -> torch.Tensor:
		x = x.to(self.device)

		if self.use_weights_after_train:
			if train_mode != TrainMode.no_train:
				self.saved_train_mode = train_mode
			train_mode = self.saved_train_mode

		for layer in self.layers:
			x = layer(x, train_mode)
		return x
