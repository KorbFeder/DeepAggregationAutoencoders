import torch 
import torch.nn as nn
import numpy as np

from archived_models.diff_edge_autoencoder import DiffEdgeAutoencoder
from archived_models.diff_edge_autoencoder import EdgeType

from typing import List, Dict, Callable

def diff_edges_visualized(model: DiffEdgeAutoencoder, use_edge_name: bool = False) -> None:
	neurons = [[{'neuron': f'{i}', 'edges': f'[normal_edge]', 'prob': [[1]]} for i in range(model.in_features)]]
	for layer in model.net:
		edge_types: List[Callable[[torch.Tensor], torch.Tensor]] = [etype for etype in EdgeType]
		if hasattr(layer, 'edge_types'):
			edge_types = layer.edge_types
		
		layer_neurons = []
		if hasattr(layer, 'prob_weights'):
			edge_type_indices = torch.nn.functional.one_hot(layer.prob_weights.argmax(-1), len(edge_types))
			probs = nn.functional.softmax(layer.prob_weights, dim=-1)
		if hasattr(layer, 'prob_edge_weights'):
			edge_type_indices = torch.nn.functional.one_hot(layer.prob_edge_weights.argmax(-1), len(edge_types))
			probs = nn.functional.softmax(layer.prob_edge_weights, dim=-1)

		if hasattr(layer, 'operators'):
			node_type_indices = torch.nn.functional.one_hot(layer.prob_node_weights.argmax(-1), len(layer.operators))
			node_probs = nn.functional.softmax(layer.prob_node_weights, dim=-1)
			nodes = node_type_indices.argmax(dim=1).tolist()
			operator = [layer.operators[index].name for index in nodes]
		else: 
			operator = layer.operator.name
	
		i = 0
		for edge_indices, prob in zip(edge_type_indices, probs):
			edges = edge_indices.argmax(dim=1).tolist()
			if use_edge_name:
				edges = [edge_types[index].name for index in edges]
			
			if hasattr(layer, 'operators'):
				layer_neurons.append({'neuron': operator[i], 'edges': edges, 'prob': prob.tolist(), 'node_prob': node_probs[i].tolist()})
			else:
				layer_neurons.append({'neuron': operator, 'edges': edges, 'prob': prob.tolist()})
			i += 1
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

		if 'node_prob' in layer[0]:
			node_probs = [str(np.round(neuron['node_prob'], 2)).replace('\n', '') for neuron in layer]
			for i in range(int((max_len - len(node_probs)) / 2)):
				node_probs.insert(i, ' ')
			node_probs += [' '] * (max_len - len(node_probs))

		connections = [str(neuron['edges']) for neuron in layer]
		for i in range(int((max_len - len(connections)) / 2)):
			connections.insert(i, ' ')
		connections += [' '] * (max_len - len(connections))

		probs = [str(np.round(neuron['prob'], 2)).replace('\n', '') for neuron in layer]

		for i in range(int((max_len - len(probs)) / 2)):
			probs.insert(i, ' ')
		probs+= [' '] * (max_len - len(probs))


		row_format = ' '.join(['{: >70}' for _ in range(max_len)])
		print(row_format.format(*operators))
		if 'node_prob' in layer[0]:
			print(row_format.format(*node_probs))
		print(row_format.format(*connections))
		print(row_format.format(*probs))
		print('\n')