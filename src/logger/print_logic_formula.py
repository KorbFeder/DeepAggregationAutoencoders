from model.edge_counting import EdgeCountingAutoencoder
from fuzzy_logic.edge_types import EdgeType
from fuzzy_logic.fuzzy_operators import T_Conorm, T_Norm
import torch
from typing import List
import math

T_NORM_NAME = "min"
T_CONORM_NAME = "max"

def hidden_layer_index(node_counting_ae: EdgeCountingAutoencoder):
	minimum = math.inf
	index = 0
	for i, layer in enumerate(node_counting_ae.layers):
		if layer.out_features < minimum:
			index = i
			minimum = layer.out_features
	return index


def print_hidden_logic_formula(node_counting_ae: EdgeCountingAutoencoder):
	print_logic_formula(node_counting_ae, hidden_layer_index(node_counting_ae), name='Hidden Logic Formula:\n')

def print_out_logic_formula(node_counting_ae: EdgeCountingAutoencoder):
	print_logic_formula(node_counting_ae, len(node_counting_ae.layers)-1, name='Output Logic Formula:\n', end_index=hidden_layer_index(node_counting_ae))

def traverse_net(node_counting_ae: EdgeCountingAutoencoder, layer_idx, node_index = 0, end_index = -1):
	if layer_idx == end_index: 
		if end_index == -1:
			return " x_" + str(node_index) + ","
		else: 
			return " h_" + str(node_index) + ","
			

	edge_counts = node_counting_ae.layers[layer_idx].edge_type_count
	equation = ""
	operator_name = f"{T_CONORM_NAME}("
	if node_counting_ae.layers[layer_idx].operator in T_Norm:
		operator_name = f"{T_NORM_NAME}("
	equation += operator_name	

	# to to child nodes
	for i, connection in enumerate(edge_counts[node_index]):
		if connection[1] > connection[0]:
			equation += traverse_net(node_counting_ae, layer_idx-1, i, end_index)

	equation += ") "
	return equation


def print_logic_formula(node_counting_ae: EdgeCountingAutoencoder, start_layer: int, name: str = "Logic Functions: \n", end_index=-1):
	layer_idx = start_layer 
	edge_counts = node_counting_ae.layers[layer_idx].edge_type_count
	equation = name
	for i, _ in enumerate(edge_counts):
		if start_layer == len(node_counting_ae.layers) - 1:
			equation += f"y_{i}: "
		else:
			equation += f"h_{i}: "
		equation += traverse_net(node_counting_ae, layer_idx, i, end_index)
		equation += "\n"

	print(equation)





