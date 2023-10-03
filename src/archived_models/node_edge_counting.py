import torch
import torch.nn as nn
import torch.nn.functional as F

from fuzzy_logic.edge_types import EdgeType
from fuzzy_logic.fuzzy_operators import T_Conorm, T_Norm

from typing import List, Union, Optional

NO_EDGE_OFFSET_T_NORM = 1
NO_EDGE_OFFSET_T_CONORM = 0

class EdgeNodeCountingLayer(nn.Module):
	def __init__(
		self: "EdgeNodeCountingLayer",
		in_features: int,
		out_features: int,
		operator_types: List[Union[T_Norm, T_Conorm]],
		edge_types: List[EdgeType],
		device: torch.device,
		seed: Optional[int] = None,
	) -> None:
		super().__init__()

		if seed != None:
			torch.manual_seed(seed)
	
		self.device = device
		self.out_features = out_features
		self.in_features = in_features
		self.edge_types = edge_types
		self.operator_types = operator_types
	
		self.edge_type_count = torch.ones(out_features, len(operator_types), in_features, len(edge_types)).to(self.device)
		self.operator_type_count = torch.ones(out_features, len(operator_types)).to(self.device)
		self.operator_indices = self.sample_operator_indices()
	
	def forward(self: "EdgeNodeCountingLayer", x: torch.Tensor) -> torch.Tensor:
		num_samples = x.shape[0]
		op_edge_selection = F.gumbel_softmax(self.edge_type_count, hard=True, dim=-1)

		edge_type_values = torch.zeros(*x.shape, len(self.edge_types)).to(self.device)
		operator_outputs = torch.zeros(num_samples, self.out_features).to(self.device)

		operators = [self.operator_types[idx] for idx in self.operator_indices]
		edge_selection = op_edge_selection[range(op_edge_selection.shape[0]), self.operator_indices]

		for node_index, operator in enumerate(operators):
			for i, edge in enumerate(self.edge_types):
				edge_type_values[..., i] = edge.value(x)

				if edge == EdgeType.no_edge:
					if operator in T_Norm:
						edge_type_values[..., i] += NO_EDGE_OFFSET_T_NORM
					if operator in T_Conorm:
						edge_type_values[..., i] += NO_EDGE_OFFSET_T_CONORM

			edge_values = torch.max(edge_selection[node_index] * edge_type_values, dim=-1).values
			operator_outputs[..., node_index] = operator.value(edge_values)
		
		return operator_outputs
	
	def sample_operator_indices(self: "EdgeNodeCountingLayer"):
		operator_selection = F.gumbel_softmax(self.operator_type_count, hard=True, dim=-1)
		return torch.max(operator_selection, dim=-1).indices.to(self.device)

	
class EdgeNodeCountingAutoencoder(nn.Module):
	def __init__( 
		self: "EdgeNodeCountingAutoencoder", 
		in_features: int,
		hidden_sizes: List[int],
		device: torch.device,
		operators: List[Union[T_Norm, T_Conorm]] = [T_Norm.min, T_Conorm.max],
		edge_types: List[EdgeType] = [EdgeType.no_edge, EdgeType.normal_edge], #, EdgeType.very, EdgeType.somewhat, EdgeType.Not],
		loss_func = nn.MSELoss(),
		seed: Optional[int] = None
	) -> None:
		super().__init__()

		self.device = device
		self.loss_func = loss_func
		self.layers: List[EdgeNodeCountingLayer] = []
		layer_sizes = [in_features, *hidden_sizes, in_features]

		for i in range(len(layer_sizes)-1):
			self.layers += [EdgeNodeCountingLayer(layer_sizes[i], layer_sizes[i + 1], operators, edge_types, device, seed)]

		self.net = nn.Sequential(*self.layers).to(self.device)
	
	def forward(self: "EdgeNodeCountingAutoencoder", x: torch.Tensor) -> torch.Tensor:
		return self.net(x)
	
	def train(self: "EdgeNodeCountingAutoencoder", x: torch.Tensor) -> torch.Tensor:
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
			layer: EdgeNodeCountingLayer = self.layers[layer_index]

			new_node_back_indices = []

			for i in range(num_batch_samples):
				new_node_back_indices.append([])

				for u, label in zip(node_back_indices[i], labels[i]):
					for op_idx, operator in enumerate(layer.operator_types):
						sorted_out, out_indices = torch.sort(output[i], 
							descending=True if operator in T_Conorm else False, dim=-1)

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
						layer.edge_type_count[u][op_idx][inc_edge_index.item()][1] += 1
						for index in dec_edge_indices:
							layer.edge_type_count[u][op_idx][index.item()][0] += 1

				# increase counters of operators depending on entropy
				for node_idx in range(layer.out_features):
					avg_entropy_per_op = torch.Tensor(len(layer.operator_types))
					for op_idx in range(len(layer.operator_types)):
						entropies = torch.Tensor(len(layer.edge_type_count[node_idx][op_idx]))
						for edge_idx, edge in enumerate(layer.edge_type_count[node_idx][op_idx]):
							distribution = torch.distributions.Categorical(edge)
							entropies[edge_idx] = distribution.entropy()
						avg_entropy_per_op[op_idx] = entropies.mean()
					layer.operator_type_count[node_idx][torch.min(avg_entropy_per_op, dim=-1).indices.item()] += 1

			self.operator_indices = layer.sample_operator_indices()

			node_back_indices = new_node_back_indices	
			layer_index -= 1
		return curr