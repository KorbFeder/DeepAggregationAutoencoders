import torch
import torch.nn as nn
import random
from enum import Enum

from typing import Callable, List, Optional

class EdgeType(Enum):
	NO_EDGE = 0
	NORMAL_EDGE = 1
	# @todo -> try out hedge operators as edges
	# VERY = 2

class EdgeSelectionLayer(nn.Module):
	def __init__(
		self: "EdgeSelectionLayer",
		in_features: int,
		out_features: int,
		operator: Callable[[torch.Tensor], torch.Tensor],
		learn_threshold: float = 0.001, 
		seed: Optional[int] = None
	) -> None:
		super().__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.prob_weights = torch.rand(out_features, in_features, len(EdgeType))
		self.operator = operator
		self.loss_function = nn.MSELoss()
		self.learn_threshold = learn_threshold

		self.seed = seed
		if seed != None:
			torch.manual_seed(seed)

	def forward(self: "EdgeSelectionLayer", x: torch.Tensor) -> torch.Tensor:
		prob = nn.functional.softmax(self.prob_weights, dim=-1)

		# @todo -> this works cause there are only 2 edge types either no edge or an normal edge, change when more 
		prob_edge = prob[..., EdgeType.NORMAL_EDGE.value]
		choices = torch.bernoulli(prob_edge)
		choices = self._zero_connection_neuron(choices)
		# since we add +1 or -1 to make values not be chosen by min max switch 0s and 1s in the choice tensor
		choices = choices.add(-1).abs()

		if self.operator == torch.max:
			choices *= -1
		
		result = torch.zeros(x.shape[0], choices.shape[0])
		for i, batch in enumerate(x):
			operator_inputs = batch.add(choices)
			result[i] = self.operator(operator_inputs, dim=-1).values

		return result

	def train(self: "EdgeSelectionLayer", x: torch.Tensor, labels: torch.Tensor, label_indices: torch.Tensor) -> torch.Tensor:
		back_label = torch.zeros(labels.shape)
		back_indices = torch.zeros(label_indices.shape)
		u = 0
		# iterate over each batch of the input batches 
		for batch_x, batch_labels, batch_label_indices in zip(x, labels, label_indices):
			# sort the array, use the montonicity of the min and max operator to find least loss value
			descending = False
			if self.operator == torch.max:
				descending = True
			values, indices = batch_x.sort(descending=descending)
		
			# iterate over the labels and label indices
			for i, (label_index, label) in enumerate(zip(batch_label_indices, batch_labels)):
				# inital loss of the first value
				last_loss = self.loss_function(values[0], label)
				last_edge = indices[0]

				neg_edges_indices = []
				neg_neuron_indices = []

				temp_edges_indices = []
				temp_neuron_indices = []
				# caneuron_indices = calculate the loss of the current label with all values until min loss reached
				for value, index in zip(values[1:], indices[1:]):
					loss = self.loss_function(value, label)
					
					if loss > last_loss:
						break
					elif loss == last_loss:
						temp_edges_indices.append(last_edge)
						temp_neuron_indices.append(label_index)

						last_edge = index
						last_loss = loss
					else:
						neg_edges_indices.append(last_edge)
						neg_neuron_indices.append(label_index)

						neg_edges_indices += temp_edges_indices
						neg_neuron_indices += temp_neuron_indices
						temp_edges_indices = []
						temp_neuron_indices = []

						last_edge = index
						last_loss = loss

				self._adjust_prob_weights(neg_edges_indices, neg_neuron_indices, False)
				self._adjust_prob_weights([last_edge] + temp_edges_indices, [label_index] + temp_neuron_indices, True)

				back_label[u][i] = label
				back_indices[u][i] = last_edge
			u += 1
		return back_label, back_indices
	
	def _zero_connection_neuron(self: "EdgeSelectionLayer", choices: torch.Tensor) -> torch.Tensor:
		connection_count = torch.count_nonzero(choices, dim=-1)
		no_conn_indices= (connection_count == 0).nonzero()
		for index in no_conn_indices:
			if self.seed != None:
				random.seed(self.seed)
			connection = random.randint(0, choices.shape[1] - 1)
			choices[index.item()][connection] += 1
		return choices 

	def _adjust_prob_weights(self: "EdgeSelectionLayer", edge_indices: List[torch.Tensor], neuron_indices: List[int], is_positive: bool) -> None:
		for edge_index, neuron_index in zip(edge_indices, neuron_indices):
			if is_positive:
				self.prob_weights[int(neuron_index)][edge_index][EdgeType.NORMAL_EDGE.value] += self.learn_threshold
				self.prob_weights[int(neuron_index)][edge_index][EdgeType.NO_EDGE.value] -= self.learn_threshold
			else:
				self.prob_weights[int(neuron_index)][edge_index][EdgeType.NORMAL_EDGE.value] -= self.learn_threshold
				self.prob_weights[int(neuron_index)][edge_index][EdgeType.NO_EDGE.value] += self.learn_threshold


class EdgeSelctionAutoencoder(nn.Module):
	def __init__(
		self: "EdgeSelctionAutoencoder",
		in_features: int,
		hidden_sizes: List[int],
		seed: Optional[int] = None
	) -> None:
		super().__init__()
		self.in_features = in_features
		layer_sizes = [in_features, *hidden_sizes, in_features]
		self.operators = [torch.min if i % 2 ==  0 else torch.max for i in range(len(layer_sizes) - 1)]
		self.layers: List[EdgeSelectionLayer] = []

		for i in range(len(layer_sizes)-1):
			self.layers += [EdgeSelectionLayer(layer_sizes[i], layer_sizes[i + 1], self.operators[i], seed=seed)]
	
	def forward(self: "EdgeSelctionAutoencoder", x: torch.Tensor, is_train: bool = False) -> torch.Tensor:
		if is_train: 
			activations = []
			for layer in self.layers:
				x = layer(x)
				activations.append(x)
			return activations
		else: 
			for layer in self.layers:
				x = layer(x)
			return x
			
	
	def train(self: "EdgeSelctionAutoencoder", x: torch.Tensor) -> torch.Tensor:
		activations = self.forward(x, is_train=True)
		y = torch.Tensor(x)
		y_indices = torch.Tensor(range(y.shape[1])).repeat(y.shape[0], 1)

		# propagate backwards
		activation_tensor = activations[:-1]
		activation_tensor.insert(0, x)
		for layer, activation in zip(reversed(self.layers), reversed(activation_tensor)):
			y, y_indices = layer.train(activation, y, y_indices)
		
		return activations[-1]


