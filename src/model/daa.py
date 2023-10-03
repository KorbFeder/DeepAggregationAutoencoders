import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

from fuzzy_logic.edge_types import EdgeType
from fuzzy_logic.fuzzy_operators import T_Conorm, T_Norm

from typing import List, Union, Optional
from enum import Enum

class NodeInitailzation(Enum):
	LAYER_WISE = 0
	ALTERNATING = 1
	RANDOM = 2

def get_neuron_configurations(operators: List[Union[T_Norm, T_Conorm]], node_initialization: NodeInitailzation, layer_sizes: List[int]):
	resulting_operators = []
	if node_initialization == NodeInitailzation.LAYER_WISE:
		for i, size in enumerate(layer_sizes):
			operator = operators[i % len(operators)]
			resulting_operators.append([operator for _ in range(size)])
	
	elif node_initialization == NodeInitailzation.ALTERNATING:
		for i, size in enumerate(layer_sizes):
			layer_op = []
			for j in range(size):
				operator = operators[j % len(operators)]
				layer_op.append(operator)
			resulting_operators.append(layer_op)
	
	else: # Random
		for i, size in enumerate(layer_sizes):
			layer_op = []
			for j in range(size):
				operator = operators[random.randint(0, len(operators)-1)]
				layer_op.append(operator)
			resulting_operators.append(layer_op)

	return resulting_operators



NO_EDGE_OFFSET_T_NORM = 2
NO_EDGE_OFFSET_T_CONORM = -1

class DAALayer(nn.Module):
	def __init__(
		self: "DAALayer",
		in_features: int,
		out_features: int,
		operators: List[Union[T_Norm, T_Conorm]],
		edge_types: List[EdgeType],
		device: torch.device,
		seed: Optional[int] = None,
	) -> None:
		super().__init__()

		if seed != None:
			torch.manual_seed(seed)
	
		self.device = device
		self.count_size = 0.1
		self.edge_type_count = torch.ones(out_features, in_features, len(edge_types)).to(self.device)
		self.out_features = out_features
		self.in_features = in_features
		self.edge_types = edge_types
		self.operators = operators
	
	def forward(self: "DAALayer", x: torch.Tensor) -> torch.Tensor:
		num_samples = x.shape[0]
		selection = F.gumbel_softmax(self.edge_type_count, hard=True, dim=-1)

		edge_type_values = torch.zeros(*x.shape, len(self.edge_types)).to(self.device)
		operator_outputs = torch.zeros(num_samples, self.out_features).to(self.device)

		for node_index, operator in enumerate(self.operators):
			for i, edge in enumerate(self.edge_types):
				edge_type_values[..., i] = edge.value(x)

				if edge == EdgeType.no_edge:
					if operator in T_Norm:
						edge_type_values[..., i] += NO_EDGE_OFFSET_T_NORM
					if operator in T_Conorm:
						edge_type_values[..., i] += NO_EDGE_OFFSET_T_CONORM
			indices = torch.max(selection[node_index], dim=-1).indices.repeat(num_samples, 1)
			edge_values = edge_type_values[torch.arange(edge_type_values.shape[0]).unsqueeze(1), torch.arange(edge_type_values.shape[1]), indices]

			operator_outputs[..., node_index] = operator.value(edge_values)

		return operator_outputs
	
class DAAutoencoder(nn.Module):
	def __init__( 
		self: "DAAutoencoder", 
		in_features: int,
		hidden_sizes: List[int],
		device: torch.device,
		operators: List[Union[T_Norm, T_Conorm]] = [T_Norm.min, T_Conorm.max],
		edge_types: List[EdgeType] = [EdgeType.no_edge, EdgeType.normal_edge], #, EdgeType.very, EdgeType.somewhat, EdgeType.Not],
		loss_func = nn.MSELoss(),
		count_step_size: int = 0.2,
		node_initialization: NodeInitailzation = NodeInitailzation.LAYER_WISE,
		seed: Optional[int] = None
	) -> None:
		super().__init__()

		self.device = device
		self.loss_func = loss_func
		self.layers: List[DAALayer] = []
		layer_sizes = [in_features, *hidden_sizes, in_features]
		self.layer_sizes = layer_sizes
		self.edge_types = edge_types
		self.count_step_size = count_step_size
		self.node_initialization = node_initialization
		self.operators = get_neuron_configurations(operators, self.node_initialization, layer_sizes[1:])

		for i in range(len(layer_sizes)-1):
			self.layers += [DAALayer(layer_sizes[i], layer_sizes[i + 1], self.operators[i], edge_types, device, seed)]

		self.net = nn.Sequential(*self.layers).to(self.device)
	
	def forward(self: "DAAutoencoder", x: torch.Tensor) -> torch.Tensor:
		return self.net(x)
	
	def train(self: "DAAutoencoder", x: torch.Tensor) -> torch.Tensor:
		x = x.to(self.device)
		labels = x
		layer_outputs = [x]
		curr = x
		num_batch_samples = len(x)

		#forward pass
		for layer in self.layers:
			curr = layer.forward(curr)
			layer_outputs.append(curr)
		
		layer_index = len(self.layers) - 1
		node_back_indices = [list(range(x.shape[1])) for _ in x]
		for output in reversed(layer_outputs[:-1]):
			layer: DAALayer = self.layers[layer_index]

			new_node_back_indices = []

			for i in range(num_batch_samples):
				new_node_back_indices.append([])

				for u, label in zip(node_back_indices[i], labels[i]):
					sorted_out, out_indices = torch.sort(output[i], descending=True if layer.operators[u] in T_Conorm else False, dim=-1)
					errors = (sorted_out - label)**2

					dec_edge_indices = []
					inc_edge_index = out_indices[0]
					best = errors[0]

					for out_i, err in zip(out_indices, errors):
						if err < best: 
							best = err
							dec_edge_indices.append(inc_edge_index)
							inc_edge_index = out_i
						if err > best:
							break

					new_node_back_indices[i].append(inc_edge_index.item())
					# increase edges and no edges 
					layer.edge_type_count[u][inc_edge_index.item()][1] += self.count_step_size
					for index in dec_edge_indices:
						layer.edge_type_count[u][index.item()][0] += self.count_step_size

			node_back_indices = new_node_back_indices	
			layer_index -= 1
		return curr