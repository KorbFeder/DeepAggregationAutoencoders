import torch
import torch.nn as nn
import numpy as np

from fuzzy_logic.edge_types import EdgeType
from fuzzy_logic.fuzzy_operators import T_Conorm, T_Norm

from typing import List, Union, Optional

NO_EDGE_OFFSET_T_NORM = 10
NO_EDGE_OFFSET_T_CONORM = -10

class ForwardForwardCoutingLayer(nn.Module):
	def __init__(
		self: "ForwardForwardCoutingLayer",
		in_features: int,
		out_features: int,
		operator_types: List[Union[T_Norm, T_Conorm]],
		edge_types: List[EdgeType],
		device: torch.device,
	) -> None:
		super().__init__()

		self.device = device
		self.out_features = out_features
		self.in_features = in_features

		self.operator_types = operator_types
		self.edge_types = edge_types

		self.edge_type_count = torch.ones(out_features, len(operator_types), in_features, len(edge_types)).to(self.device)
		self.operator_type_counts = torch.ones(out_features, len(operator_types)).to(self.device)
		self.operator_indices = self.sample_operator_indices()

	def forward(self: "ForwardForwardCoutingLayer", x: torch.Tensor) -> torch.Tensor:
		num_samples = x.shape[0]
		node_values = torch.zeros(num_samples, self.out_features).to(self.device)

		# loop over every sample in the  batch
		for sample_index in range(num_samples):
			# loop over every node of the layer
			for node_index in range(self.out_features):
				sample = x[sample_index]

				# sample operator and edge
				edge_type_indices = torch.multinomial(
					self.edge_type_count[node_index][self.operator_indices[node_index]], 
					num_samples=1, 
					replacement=True
				).squeeze(-1).to(torch.int)

				# check if every node has at least an input connection, if it hasn't give it one at random
				if torch.count_nonzero(edge_type_indices) == 0:
					edge_type_indices[torch.randint(len(edge_type_indices), size=(1,))] = 1

				edge_values = torch.zeros(self.in_features).to(self.device)

				operator = self.operator_types[self.operator_indices[node_index]]

				# loop over every edge
				for edge_index, edge_type_index in enumerate(edge_type_indices):
					# get the value of each edge depending on the type 
					edge_values[edge_index] = self.edge_types[edge_type_index].value(sample[edge_index])

					# check for the type of operator and deside what value to use
					if self.edge_types[edge_type_index] == EdgeType.no_edge:
						if operator in T_Norm:
							edge_values[edge_index] += NO_EDGE_OFFSET_T_NORM
						if operator in T_Conorm:
							edge_values[edge_index] += NO_EDGE_OFFSET_T_CONORM

				# aggregate the edges with the operator of the corresponding node
				node_value = operator.value(edge_values)
				node_values[sample_index][node_index] = node_value

		return node_values
	
	def sample_operator_indices(self: "ForwardForwardCoutingLayer"):
		operator_indices = []
		for node_index in range(self.out_features):
			operator_type_index = torch.multinomial(
				self.operator_type_counts[node_index],
				num_samples=1,
				replacement=True
			).squeeze(-1).to(torch.int)
			operator_indices.append(operator_type_index)
		return operator_indices


class ForwardForwardNodeEdgeCoutingAutoencoder(nn.Module):
	def __init__(self: "ForwardForwardNodeEdgeCoutingAutoencoder",
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
			np.random.seed(seed)

		self.device = device
		self.loss_func = loss_func
		self.layers: List[ForwardForwardCoutingLayer] = []
		layer_sizes = [in_features, *hidden_sizes, in_features]

		for i in range(len(layer_sizes)-1):
			self.layers += [ForwardForwardCoutingLayer(layer_sizes[i], layer_sizes[i + 1], operators, edge_types, device)]

		self.net = nn.Sequential(*self.layers).to(self.device)
	
	def forward(self: "ForwardForwardNodeEdgeCoutingAutoencoder", x: torch.Tensor) -> torch.Tensor:
		return self.net(x)

	def train(self: "ForwardForwardNodeEdgeCoutingAutoencoder", x: torch.Tensor):
		num_samples = x.shape[0]
		layer_outputs = [x]
		labels = x
		curr = x
		label_node_indices = [list(range(len(_x))) for _x in x]
		
		# first forward pass
		for layer in self.layers:
			curr = layer(curr)
			layer_outputs.append(curr)

		# the last layer has to have the target labels
		layer_outputs[-1] = x

		# second forward pass	
		for layer_index in range(len(layer_outputs)-1):
			layer: ForwardForwardCoutingLayer = self.layers[layer_index]
			
			for sample_index in range(num_samples):
				layer_input = layer_outputs[layer_index][sample_index]
				layer_output = layer_outputs[layer_index + 1][sample_index]

				new_label_node_indices = []
				update_edge_type_count = torch.zeros(*layer.edge_type_count.shape).to(self.device)

				for label_node_index, label in zip(label_node_indices[sample_index], labels[sample_index]):
					# select the node with the smallest error 
					errors: torch.Tensor = (layer_output - label) ** 2
					chosen_next_node_indices,  = torch.where(errors == errors.min())
					chosen_next_node_index = np.random.choice(chosen_next_node_indices.cpu())

					new_label_node_indices.append(chosen_next_node_index)

					# for each operator type of that node increment the connection
					for op_idx in range(len(layer.operator_types)):
						update_edge_type_count[chosen_next_node_index][op_idx][label_node_index][1] += 1
						#layer.edge_type_count[chosen_next_node_index][op_idx][label_node_index][1] += 1

					# increment no edge for edges that would screw with the desired value
					for layer_in_index, layer_in_value in enumerate(layer_input):
						for op_idx, operator in enumerate(layer.operator_types):
							if operator in T_Conorm:
								if layer_in_value > label and layer_in_index != label_node_index:
									update_edge_type_count[chosen_next_node_index][op_idx][layer_in_index][0] += 1
									#layer.edge_type_count[chosen_next_node_index][op_idx][layer_in_index][0] += 1
							if operator in T_Norm:
								if layer_in_value < label and layer_in_index != label_node_index:
									update_edge_type_count[chosen_next_node_index][op_idx][layer_in_index][0] += 1
									#layer.edge_type_count[chosen_next_node_index][op_idx][layer_in_index][0] += 1

				label_node_indices[sample_index] = new_label_node_indices
				# if an edge has an increment to edge and no edge remove increment to no edge
				for per_node in update_edge_type_count:
					for per_operator in per_node:
						for per_edge in per_operator:
							if per_edge[1] > 0:
								per_edge[0] = 0

				layer.edge_type_count += update_edge_type_count

		return curr