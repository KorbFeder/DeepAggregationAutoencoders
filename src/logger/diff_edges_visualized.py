import torch 
import torch.nn as nn
import numpy as np

from model.diff_edge_autoencoder import DiffEdgeAutoencoder
from model.diff_edge_autoencoder import EdgeType

from typing import List, Dict, Callable

def diff_edges_visualized(model: DiffEdgeAutoencoder, use_edge_name: bool = False) -> None:
	neurons = [[{'neuron': f'{i}', 'edges': f'[normal_edge]', 'prob': [[1]]} for i in range(model.in_features)]]
	for layer in model.net:
		edge_types: List[Callable[[torch.Tensor], torch.Tensor]] = [etype for etype in EdgeType]
		layer_neurons = []
		edge_type_indices = torch.nn.functional.one_hot(layer.prob_weights.argmax(-1), len(edge_types))
		probs = nn.functional.softmax(layer.prob_weights, dim=-1)
		for indices, prob in zip(edge_type_indices, probs):
			edges = indices.argmax(dim=1).tolist()
			if use_edge_name:
				edges = [edge_types[index].name for index in edges]
			layer_neurons.append({'neuron': layer.operator.__name__, 'edges': edges, 'prob': prob.tolist()})
		neurons.append(layer_neurons)
	_print_table(neurons)

def _print_table(table_data: List[List[Dict]]) -> None:
	max_len = 0
	for layer in table_data:
		if len(layer) > max_len:
			max_len = len(layer)

	for layer in table_data:
		operators = [neuron['neuron'] for neuron in layer]
		for i in range(int((max_len - len(operators)) / 2)):
			operators.insert(i, ' ')
		operators += [' '] * (max_len - len(operators))

		connections = [str(neuron['edges']) for neuron in layer]
		for i in range(int((max_len - len(connections)) / 2)):
			connections.insert(i, ' ')
		connections += [' '] * (max_len - len(connections))

		probs = [str(np.round(neuron['prob'], 2)).replace('\n', '') for neuron in layer]

		for i in range(int((max_len - len(probs)) / 2)):
			probs.insert(i, ' ')
		probs+= [' '] * (max_len - len(probs))


		row_format = ' '.join(['{: >30}' for _ in range(max_len)])
		print(row_format.format(*operators))
		print(row_format.format(*connections))
		print(row_format.format(*probs))
		print('\n')