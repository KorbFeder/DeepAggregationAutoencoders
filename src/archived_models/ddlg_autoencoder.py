import torch
import torch.nn as nn

from typing import List, Callable
from fuzzy_logic.fuzzy_operators import fuzzy_max, fuzzy_min, fuzzy_coalg, fuzzy_alg, fuzzy_ein, fuzzy_coein, fuzzy_luk, fuzzy_coluk

def create_function_table():
	logic_func: List[Callable[[torch.Tensor], torch.Tensor]] = [
		fuzzy_min,
		fuzzy_max,
		#fuzzy_alg,
		#fuzzy_coalg,
		fuzzy_ein,
		fuzzy_coein,
		#fuzzy_luk,
		#fuzzy_coluk
	]
	return logic_func
	

class DdlgLayer(nn.Module):
	def __init__(
		self: "DdlgLayer",
		in_features: int,
		out_features: int,
		num_connections: int,
		device: torch.device,
	) -> None:
		super().__init__()

		assert num_connections <= in_features

		self.is_train = False
		self.device = device
		self.logic_functions = create_function_table()
		self.weights = nn.parameter.Parameter(torch.randn(out_features, len(self.logic_functions), device=device))
		self.in_features = in_features
		self.out_features = out_features

		# creates indices on the inputs that are connected to each node
		self.connection_indices = torch.zeros(out_features, num_connections).to(device)
		for i in range(len(self.connection_indices)): 
			self.connection_indices[i] = torch.randperm(in_features)[:num_connections]

	def forward(self: "DdlgLayer", x: torch.Tensor):
		# only use the chosen connections 
		out_indices = torch.arange(self.out_features).to(self.device)
		conn_indices = self.connection_indices[out_indices].cpu().numpy()
		features = x[:, conn_indices].to(self.device)  

		if self.is_train:
			prob = nn.functional.softmax(self.weights, dim=-1)
		else: 
			prob = torch.nn.functional.one_hot(self.weights.argmax(-1), len(self.logic_functions)).to(torch.float32)

		result = torch.zeros((len(x), self.out_features)).to(self.device)
		for i, op in enumerate(self.logic_functions):
			op_result = op(features, dim=-1)
			result = result + prob[..., i] * op_result
		return result

class DdlgAutoencoder(nn.Module):
	def __init__(
		self: "DdlgAutoencoder",
		in_features: int,
		hidden_sizes: List[int],
		num_connections: int,
		device: torch.device,
	
	) -> None:
		super().__init__()
		self.device = device
		self.in_features = in_features
		layer_sizes = [in_features, *hidden_sizes, in_features]
		layers = []

		for i in range(len(layer_sizes)-1):
			layers += [DdlgLayer(layer_sizes[i], layer_sizes[i + 1], num_connections, device)]

		self.net = nn.Sequential(*layers).to(self.device)
	
	def forward(self: "DdlgAutoencoder", x: torch.Tensor, is_train = False)-> torch.Tensor:
		for layer in self.net:
			layer.is_train = is_train
		return self.net(x.to(self.device))
