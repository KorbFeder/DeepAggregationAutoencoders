import torch 
import torch.nn as nn

from archived_models.ddlg_autoencoder import DdlgAutoencoder

from typing import List, Dict

def ddlg_neurons(model: DdlgAutoencoder) -> None:
	neurons = [[{'name': f'{i}', 'connections': f'[{i}]'} for i in range(model.in_features)]]
	for layer in model.net:
		layer_neurons = []
		operator_indices = torch.nn.functional.one_hot(layer.weights.argmax(-1), len(layer.logic_functions))
		for i, indices in enumerate(operator_indices):
			operator = layer.logic_functions[indices.argmax().item()]
			layer_neurons.append({'name': operator.__name__, 'connections': layer.connection_indices[i].cpu().tolist()})
		neurons.append(layer_neurons)
	_print_table(neurons)

def _print_table(table_data: List[List[Dict]]) -> None:
	max_len = 0
	for layer in table_data:
		if len(layer) > max_len:
			max_len = len(layer)

	for layer in table_data:
		operators = [neuron['name'] for neuron in layer]
		for i in range(int((max_len - len(operators)) / 2)):
			operators.insert(i, ' ')
		operators += [' '] * (max_len - len(operators))

		connections = [str(neuron['connections']) for neuron in layer]
		for i in range(int((max_len - len(connections)) / 2)):
			connections.insert(i, ' ')
		connections += [' '] * (max_len - len(connections))


		row_format = ' '.join(['{: >20}' for _ in range(max_len)])
		print(row_format.format(*operators))
		print(row_format.format(*connections))
		print('\n')