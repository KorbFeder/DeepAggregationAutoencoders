import torch
import torch.nn as nn

from typing import List, Callable
from fuzzy_logic.fuzzy_operators import fuzzy_max, fuzzy_min

def create_function_table():
	logic_func: List[Callable[[torch.Tensor], torch.Tensor]] = [
		fuzzy_min,
		fuzzy_max
	]
	return logic_func
	

class DdlgLayer(nn.Module):
	def __init__(
		self: "DdlgLayer",
		in_features: int,
		out_features: int,
		num_logic_functions: int,
		num_connection: int,
		device: torch.device,
	) -> None:
		super().__init__()

		assert num_connection < in_features

		self.device = device
		self.weights = nn.parameter.Parameter(torch.randn(num_logic_functions))
		self.in_dim = in_features
		self.out_dim = out_features
		self.logic_functions = create_function_table()

		# creates indices on the inputs that are connected to each node
		self.connections = torch.zeros(in_features, out_features)
		for connection in self.connections:
			connection[torch.randperm(in_features)[:num_connection]] = 1

	def forward(self: "DdlgLayer", x: torch.Tensor):
		# only use the chosen connections 
		x = torch.mul(self.connections, x)
		prob = nn.Softmax(self.weights)





class DdlgAutoencoder(nn.Module):
	def __init__(
		self: "DdlgAutoencoder",
		in_features: int,
		hidden_sizes: List[int],
		device: torch.device,
		activation: nn.Module = nn.ReLU(),
		out_activation: nn.Module = nn.Sigmoid()
	
	) -> None:
		super().__init__()

DdlgLayer(10, 10, 3, 3, None)