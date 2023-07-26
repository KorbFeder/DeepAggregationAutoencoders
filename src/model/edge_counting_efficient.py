import torch
import torch.nn as nn
import torch.nn.functional as F

from fuzzy_logic.edge_types import EdgeType
from fuzzy_logic.fuzzy_operators import T_Conorm, T_Norm

from typing import List, Union, Optional
from dataclasses import dataclass

@dataclass
class BackprobValue:
	prev_node_indices: List[List[int]]
	values: torch.Tensor

NO_EDGE_OFFSET_T_NORM = 1
NO_EDGE_OFFSET_T_CONORM = 0

class EdgeCountingLayer(nn.Module):
	def __init__(
		self: "EdgeCountingLayer",
		in_features: int,
		out_features: int,
		operator: Union[T_Norm, T_Conorm],
		edge_types: List[EdgeType],
		device: torch.device,
		seed: Optional[int] = None,
	) -> None:
		super().__init__()

		if seed != None:
			torch.manual_seed(seed)
	
		self.device = device
		self.edge_type_count = torch.zeros(out_features, in_features, len(edge_types)).to(self.device)
		self.out_features = out_features
		self.in_features = in_features
		self.edge_types = edge_types
		self.operator = operator
	
	def forward(self: "EdgeCountingLayer", x: torch.Tensor) -> torch.Tensor:
		num_samples = x.shape[0]
		selection = F.gumbel_softmax(self.edge_type_count, hard=True, dim=-1)

		edge_type_values = torch.zeros(*x.shape, len(self.edge_types)).to(self.device)

		for i, edge in enumerate(self.edge_types):
			edge_type_values[..., i] = edge.value(x)

			if edge == EdgeType.no_edge:
				if self.operator in T_Norm:
					edge_type_values[..., i] += NO_EDGE_OFFSET_T_NORM
				if self.operator in T_Conorm:
					edge_type_values[..., i] += NO_EDGE_OFFSET_T_CONORM

		each_sample_selection = selection.unsqueeze(0).repeat(num_samples, 1, 1, 1)
		out_edge_values = edge_type_values.unsqueeze(1).repeat(1, self.out_features, 1, 1)
		
		edge_values = torch.max(each_sample_selection * out_edge_values, dim=-1).values
		
		return self.operator.value(edge_values)

	def train(self: "EdgeCountingLayer", forward_values: torch.Tensor, label: BackprobValue):
		is_descending = False
		if self.operator in T_Conorm:
			is_descending = True

		sorted_out, out_indices = torch.sort(forward_values, descending=is_descending, dim=-1)

		repeated_out = sorted_out.unsqueeze(1).repeat(1, label.values.shape[-1], 1)	
		repeated_label = label.values.unsqueeze(-1).repeat(1, 1, sorted_out.shape[-1])	

		errors = (repeated_out - repeated_label)**2
		_, indices = torch.min(errors, dim=-1)

		inc_indices = torch.take(out_indices, indices)
		dec_indices = list([list(list(range(i)) for i in _indices if (i-1) >= 0) for _indices in indices])

		self.edge_type_count[label.prev_node_indices][inc_indices][1] += 1
		self.edge_type_count[label.prev_node_indices][dec_indices][0] += 1

		label.prev_node_indices = inc_indices

		print(errors)
	
class EdgeCountingAutoencoder(nn.Module):
	def __init__( 
		self: "EdgeCountingAutoencoder", 
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
		self.layers: List[EdgeCountingLayer] = []
		layer_sizes = [in_features, *hidden_sizes, in_features]
		self.operators = (operators * int((len(layer_sizes) / len(operators)) + 1))[:len(layer_sizes)]

		for i in range(len(layer_sizes)-1):
			self.layers += [EdgeCountingLayer(layer_sizes[i], layer_sizes[i + 1], self.operators[i], edge_types, device, seed)]

		self.net = nn.Sequential(*self.layers).to(self.device)
	
	def forward(self: "EdgeCountingAutoencoder", x: torch.Tensor) -> torch.Tensor:
		return self.net(x)
	
	def train(self: "EdgeCountingAutoencoder", x: torch.Tensor) -> torch.Tensor:
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
			
			layer: EdgeCountingLayer = self.layers[layer_index].train(output, BackprobValue(node_back_indices, labels))

			is_descending = False
			if layer.operator in T_Conorm:
				is_descending = True
	
			new_node_back_indices = []

			for i in range(num_batch_samples):
				new_node_back_indices.append([])

				for u, label in zip(node_back_indices[i], labels[i]):
					sorted_out, out_indices = torch.sort(output[i], descending=is_descending, dim=-1)
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
					layer.edge_type_count[u][inc_edge_index.item()][1] += 1
					for index in dec_edge_indices:
						layer.edge_type_count[u][index.item()][0] += 1

			node_back_indices = new_node_back_indices	
			layer_index -= 1
		return curr