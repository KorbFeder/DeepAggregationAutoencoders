import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from fuzzy_logic.edge_types import EdgeType
from fuzzy_logic.fuzzy_operators import T_Conorm, T_Norm

from typing import List, Union, Optional

NO_EDGE_OFFSET_T_NORM = 1
NO_EDGE_OFFSET_T_CONORM = 0

class NodeCountingLayer(nn.Module):
	def __init__(
		self: "NodeCountingLayer",
		in_features: int,
		out_features: int,
		operators: List[Union [T_Norm, T_Conorm]],
		edge_types: List[List[EdgeType]],
		device: torch.device,
		seed: Optional[int] = None,
	) -> None:
		super().__init__()

		if seed != None:
			torch.manual_seed(seed)
	
		self.device: torch.device = device
		self.node_type_count: torch.Tensor = torch.zeros(out_features, len(operators)).to(self.device)
		self.out_features: int = out_features
		self.in_features: int = in_features
		self.edge_types: List[List[EdgeType]] = edge_types
		self.operators: List[Union [T_Norm, T_Conorm]] = operators

	def forward_every_op_result(self: "NodeCountingLayer", x: torch.Tensor):
		num_samples = x.shape[0]

		node_type_values: torch.Tensor = torch.zeros(num_samples, self.out_features, len(self.operators)).to(self.device)
		edge_values: torch.Tensor = torch.zeros(num_samples, self.out_features, self.in_features).to(self.device)

		for node_index in range(self.out_features):
			for i, edge_type in enumerate(self.edge_types[node_index]):
				value = edge_type.value(x[..., i])
				if edge_type == EdgeType.no_edge: 
					value -= 1
				edge_values[..., node_index, i] = value

		_edge_values = torch.clone(edge_values)

		prev_node_indices = []
		for n in range(num_samples):
			edge_values_per_sample = _edge_values[n]
			prev_node_index_per_sample = []
			for node_index in range(self.out_features):
				prev_node_index_per_op = []
				for i, operator in enumerate(self.operators):
					if operator in T_Norm:
						edge_values_per_sample[edge_values_per_sample == -1] = NO_EDGE_OFFSET_T_NORM
					if operator in T_Conorm:
						edge_values_per_sample[edge_values_per_sample == -1] = NO_EDGE_OFFSET_T_CONORM

					op_result = operator.value(edge_values_per_sample[node_index])
					edges = edge_values_per_sample[node_index]
					node_type_values[n, node_index, i] = op_result

					prev_node_index_per_op.append(torch.where(op_result == edges)[0][0].item())

					_edge_values = torch.clone(edge_values)
					edge_values_per_sample = _edge_values[n]

				prev_node_index_per_sample.append(prev_node_index_per_op)

			prev_node_indices.append(prev_node_index_per_sample)

		return node_type_values, prev_node_indices

	def forward_with_indices(self: "NodeCountingLayer", x: torch.Tensor) -> torch.Tensor:
		num_samples = x.shape[0]
		selection: torch.Tensor = F.gumbel_softmax(self.node_type_count, hard=True, dim=-1)
		each_sample_selection = selection.repeat(num_samples, 1, 1)

		node_type_values, indices_per_op = self.forward_every_op_result(x)
		node_values = torch.max(each_sample_selection * node_type_values, dim=-1).values
		indices = torch.max(each_sample_selection * torch.Tensor(indices_per_op).to(self.device), dim=-1).values
		
		return node_values, indices


	def forward(self: "NodeCountingLayer", x: torch.Tensor) -> torch.Tensor:
		num_samples = x.shape[0]
		selection: torch.Tensor = F.gumbel_softmax(self.node_type_count, hard=True, dim=-1)
		each_sample_selection = selection.repeat(num_samples, 1, 1)

		node_type_values, _ = self.forward_every_op_result(x)
		node_values = torch.max(each_sample_selection * node_type_values, dim=-1).values
		
		return node_values
	
class NodeCountingAutoencoder(nn.Module):
	def __init__( 
		self: "NodeCountingAutoencoder", 
		in_features: int,
		hidden_sizes: List[int],
		device: torch.device,
		operators: List[Union[T_Norm, T_Conorm]] = [T_Norm.min, T_Conorm.max],
		edge_types: List[EdgeType] = [EdgeType.no_edge, EdgeType.normal_edge], #, EdgeType.very, EdgeType.somewhat, EdgeType.Not],
		loss_func = nn.MSELoss(),
		seed: Optional[int] = None
	) -> None:
		super().__init__()
		if seed != None:
			torch.manual_seed(seed)
			random.seed(seed)

		self.device = device
		self.loss_func = loss_func
		self.layers: List[NodeCountingLayer] = []
		layer_sizes = [in_features, *hidden_sizes, in_features]
		self.operators = operators

		def random_edges(in_size, out_size):
			_edge_types = []
			for _ in range(out_size): 
				t = []	
				for _ in range(in_size):
					index_type = random.randint(0, len(edge_types)-1)
					t.append(edge_types[index_type])
				
				if t.count(EdgeType.no_edge) == len(t):
					t[random.randint(0, len(t)-1)] = EdgeType.normal_edge

				_edge_types.append(t)
			_edge_types = np.array(_edge_types)
			for i_col in range(_edge_types.shape[1]):
				col = _edge_types[:, i_col]
				if len(col[col == EdgeType.no_edge]) == len(col):
					col[random.randint(0, len(col)-1)] = EdgeType.normal_edge

			return _edge_types

		for i in range(len(layer_sizes)-1):
			self.layers += [NodeCountingLayer(layer_sizes[i], layer_sizes[i + 1], self.operators, random_edges(layer_sizes[i], layer_sizes[i + 1]), device, seed)]

		self.net = nn.Sequential(*self.layers).to(self.device)
	
	def forward(self: "NodeCountingAutoencoder", x: torch.Tensor) -> torch.Tensor:
		return self.net(x)
	
	def train(self: "NodeCountingAutoencoder", x: torch.Tensor) -> torch.Tensor:
		x = x.to(self.device)
		labels = x
		layer_outputs = [x]
		indices_outputs = []
		curr = x
		num_batch_samples = len(x)

		#forward pass
		for layer in self.layers:
			curr, prev_indices = layer.forward_with_indices(curr)
			layer_outputs.append(curr)
			indices_outputs.append(prev_indices)
		
		layer_index = len(self.layers) - 1
		node_back_indices = [list(range(x.shape[1])) for _ in x]

		#for output in reversed(layer_outputs[:-1]):
		#	layer: NodeCountingLayer = self.layers[layer_index]
		#	operators_outputs = layer.forward_every_op_result(output)

		#	error = (operators_outputs - labels.unsqueeze(2).repeat(1, 1, len(layer.operators))) ** 2

		#	count_indices = torch.min(error, dim=-1).indices
		#	for indices in count_indices:
		#		for i, _ in enumerate(layer.node_type_count):
		#			layer.node_type_count[i][indices[i]] += 1

		#	layer_index -= 1
		for output, node_indices in reversed(list(zip(layer_outputs[:-1], indices_outputs))):
			layer: NodeCountingLayer = self.layers[layer_index]
			operators_outputs, _ = layer.forward_every_op_result(output)
			for i in range(num_batch_samples):
				for u, label in zip(node_back_indices[i], labels[i]):

					error = (operators_outputs[i][int(u)] - label) ** 2

					#count_index = torch.min(error, dim=-1).indices
					count_index = (error == torch.min(error)).nonzero()
					if len(count_index) == 1:
						layer.node_type_count[int(u)][count_index.item()] += 1
			node_back_indices = node_indices.tolist()
			layer_index -= 1
		return curr
