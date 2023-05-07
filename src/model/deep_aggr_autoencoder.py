import torch
import torch.nn as nn
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import DataLoader

from typing import List, Callable, Tuple

def fuzzy_min(x: torch.Tensor, dim: int=0):
	return torch.min(x, dim=dim).values

def fuzzy_max(x: torch.Tensor, dim: int=0):
	return torch.max(x, dim=dim).values

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
		num_connections: int,
		operator_table: List[Callable[[torch.Tensor, int], torch.Tensor]]
	) -> None:
		super().__init__()

		self.out_features = out_features
		self.in_features = in_features
		self.operator_table = operator_table

		# table of indices that gives each neuron an index to an operator from the operator table
		self.operator_table_indices = [random.randint(0, len(self.operator_table) - 1) for _ in range(out_features)]

		# creates indices on the inputs that are connected to each node
		self.connection_indices = torch.zeros(out_features, num_connections)
		for i in range(len(self.connection_indices)): 
			self.connection_indices[i] = torch.randperm(in_features)[:num_connections]

	def forward(self: "DeepAggregateLayer", x: torch.Tensor, is_train: bool = False) -> torch.Tensor:
		if is_train:
			return self._forward_train(x)

		# @Todo -> maybe make this more efficient
		batch_size = x.shape[0]
		result = torch.zeros(batch_size, self.out_features)

		for i in range(self.out_features):
			operator_index = self.operator_table_indices[i]
			operator = self.operator_table[operator_index]
			for u in range(batch_size):
				result[u][i] = operator(x[u][self.connection_indices[i].numpy()])
	
		return result
	
	def _forward_train(self: "DeepAggregateLayer", x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		out_indices = torch.arange(self.out_features)
		conn_indices = self.connection_indices[out_indices].numpy()
		features = x[:, conn_indices]  
		output = torch.stack([operator(features, dim=2) for operator in self.operator_table], dim=1)
		fwd = output.transpose(1, 2)[:, range(output.shape[2]), self.operator_table_indices]
		return fwd, output

class DeepAggregateAutoEncoder(nn.Module):
	def __init__(
		self: "DeepAggregateAutoEncoder",
		in_features: int,
		hidden_sizes: List[int],
		num_connections_per_layer: List[int]
	) -> None:
		super().__init__()
		layer_sizes = [in_features, *hidden_sizes]
		self.layers = []
		self.num_hidden_neurons = sum([*hidden_sizes])
		self.operator_table = create_operator_table()

		for i in range(len(layer_sizes)-1):
			self.layers += [DeepAggregateLayer(layer_sizes[i], layer_sizes[i+1], num_connections_per_layer[i], self.operator_table)]
		
		self.output_layer = DeepAggregateLayer(layer_sizes[-1], in_features, num_connections_per_layer[-1], self.operator_table)

	def forward(self: "DeepAggregateAutoEncoder", x: torch.Tensor, is_train: bool = False) -> torch.Tensor:
		if is_train:
			return self._forward_train(x)

		for layer in self.layers:
			x = layer(x, is_train)

		return self.output_layer(x)

	def _forward_train(self: "DeepAggregateAutoEncoder", x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		layer_activations = torch.tensor(len(self.layers), )
		layer_activations = []

		for layer in self.layers:
			x, activation = layer(x, True)
			layer_activations.append(activation)

		output, output_activation = self.output_layer(x, True)

		layer_activations.append(output_activation)
		return output, np.array(layer_activations)

	def train(self: "DeepAggregateAutoEncoder", x: torch.Tensor) -> Tuple[float, float]:
		output, target_activation = self.forward(x, True)
		_, prediction_activation = self.forward(output, True)

		hidden_loss = ((target_activation[:-1] - prediction_activation[:-1]) ** 2)
		output_loss = ((target_activation[-1] - np.repeat(x[:, np.newaxis, :], target_activation[-1].shape[1], axis=1)) ** 2)

		for loss, layer in zip(hidden_loss, self.layers):
			indices = torch.argmin(loss, dim=1)
			indices_occurrences = indices.mode(dim=0).values
			layer.operator_table_indices = indices_occurrences.tolist()

		indices = torch.argmin(output_loss, dim=1)
		indices_occurrences = indices.mode(dim=0).values
		self.output_layer.operator_table_indices = indices_occurrences.tolist()
		
		return output_loss.sum().item(), hidden_loss.sum().sum().item()

