import torch
import torch.nn as nn
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import DataLoader

from typing import List

#def null_operator(x: torch.Tensor):
#	return torch.zeros(1)

def fuzzy_min(x: torch.Tensor):
	return torch.min(x)

def fuzzy_max(x: torch.Tensor):
	return torch.max(x)

def create_operator_table():
	return [
		#null_operator, 
		fuzzy_min,
		fuzzy_max,
	]

class DeepAggregateLayer(nn.Module):
	def __init__(
		self: "DeepAggregateLayer",
		in_features: int, 
		out_features: int, 
		num_connections: int
	) -> None:
		super().__init__()

		self.out_features = out_features
		self.in_features = in_features
		self.operator_table = create_operator_table()

		# table of indices that gives each neuron an index to an operator from the operator table
		self.operator_table_indices = [random.randint(0, len(self.operator_table) - 1) for _ in range(out_features)]

		# creates indices on the inputs that are connected to each node
		self.connection_indices = torch.zeros(out_features, num_connections)
		for i in range(len(self.connection_indices)): 
			self.connection_indices[i] = torch.randperm(in_features)[:num_connections]

	def forward(self: "DeepAggregateLayer", x: torch.Tensor, is_train: bool = False) -> torch.Tensor:
		if is_train:
			return self._forward_train(x)

		result = torch.zeros(self.out_features)

		for i in range(self.out_features):
			operator_index = self.operator_table_indices[i]
			operator = self.operator_table[operator_index]
			result[i] = operator(x[self.connection_indices[i].numpy()])
		return result
	
	def _forward_train(self: "DeepAggregateLayer", x: torch.Tensor) -> torch.Tensor:
		result = torch.zeros(self.out_features, len(self.operator_table))

		for i in range(self.out_features):
			for u, operator in enumerate(self.operator_table):
				result[i][u] = operator(x[self.connection_indices[i].numpy()])
		return result


class DeepAggregateAutoEncoder(nn.Module):
	def __init__(
		self: "DeepAggregateAutoEncoder",
		in_features: int,
		hidden_sizes: List[int],
		num_connections_per_layer: List[int]
	) -> None:
		super().__init__()
		layer_sizes = [in_features, *hidden_sizes, in_features]
		self.layers = []
		self.num_hidden_neurons = sum(hidden_sizes)

		for i in range(len(layer_sizes)-1):
			self.layers += [DeepAggregateLayer(layer_sizes[i], layer_sizes[i+1], num_connections_per_layer[i])]

	def forward(self: "DeepAggregateAutoEncoder", x: torch.Tensor):
		activations = torch.Tensor(x.shape[0], self.num_hidden_neurons)
		i = 0

		for layer in self.layers:
			x = layer(x)

			new_i = x.shape[1] + i
			activations[:, i, new_i] = x
			i = new_i

		return x, activations

class DeepAggregateTrainer:
	def __init__(
		self: "DeepAggregateTrainer",
		model: nn.Module,
		data_loader: DataLoader
	) -> None:
		self.model = model
		self.data_loader = data_loader

	def train(self: "DeepAggregateTrainer") -> None: 
		for x, _ in tqdm(self.data_loader):
			output, target_activation = self.model(x)
			_, prediction_activation = self.model(output)

			loss = ((target_activation - prediction_activation) ** 2)


