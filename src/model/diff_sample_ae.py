import torch
import torch.nn as nn
import torch.nn.functional as F
from fuzzy_logic.fuzzy_operators import T_Norm, T_Conorm
from functools import partial
from enum import Enum

from fuzzy_logic.edge_types import EdgeType

from typing import List, Union

NO_EDGE_OFFSET_T_NORM = 1
NO_EDGE_OFFSET_T_CONORM = 0

class DiffMin(torch.autograd.Function):
	@staticmethod
	def forward(ctx, input):
		ctx.save_for_backward(input)
		return T_Norm.min(input)

	@staticmethod
	def backward(ctx, input):
		input, = ctx.saved_tensors


class DiffSampleLayer(nn.Module):
	def __init__(
		self: "DiffSampleLayer",
		in_features: int,
		out_features: int,
		device: torch.device,
		operator: Union[T_Norm, T_Conorm],
		edge_types: List[EdgeType],
	) -> None:
		super().__init__()

		torch.manual_seed(0)
		self.edge_types = edge_types
		self.device = device
		self.operator = operator
		self.edges = nn.parameter.Parameter(torch.randn(out_features, in_features, len(edge_types), device=self.device), requires_grad=True)
	
		self.in_features: int = in_features
		self.out_features: int = out_features

	def forward(self: "DiffSampleLayer", x: torch.Tensor) -> torch.Tensor:
		num_samples = x.shape[0]
		selection = F.gumbel_softmax(self.edges, hard=True, dim=-1)

		edge_type_values = torch.zeros(*x.shape, len(self.edge_types)).to(self.device)
		for i, edge in enumerate(self.edge_types):
			edge_type_values[..., i] = edge.value(x)

			if edge == EdgeType.no_edge:
				if self.operator in T_Norm:
					edge_type_values[..., i] += NO_EDGE_OFFSET_T_NORM
				if self.operator in T_Conorm:
					edge_type_values[..., i] += NO_EDGE_OFFSET_T_CONORM

		each_sample_selection = selection.unsqueeze(0).repeat(num_samples, 1, 1, 1)
		out_edge_values = edge_type_values.unsqueeze(1).repeat(1, self.out_features, 1, 1)
		
		edge_values = torch.max(each_sample_selection * out_edge_values, dim=-1).values
		
		val = self.operator.value(edge_values)
		return val

class DiffSampleAutoencoder(nn.Module):
	def __init__(
		self: "DiffSampleAutoencoder", 
		in_features: int,
		hidden_sizes: List[int],
		device: torch.device,
		operators: List[Union[T_Norm, T_Conorm]] = [T_Norm.min, T_Conorm.max],
		edge_types: List[EdgeType] = [EdgeType.no_edge, EdgeType.normal_edge], #, EdgeType.very, EdgeType.somewhat, EdgeType.Not],
	) -> None:
		super().__init__()
		self.device = device
		self.in_features = in_features
		layer_sizes = [in_features, *hidden_sizes, in_features]
		layers = []

		for i in range(len(layer_sizes)-1):
			layers += [DiffSampleLayer(layer_sizes[i], layer_sizes[i + 1], device, operators[i], edge_types)]

		self.net = nn.Sequential(*layers).to(self.device)
	
	def forward(self: "DiffSampleAutoencoder", x: torch.Tensor):
		x = x.to(self.device)
		return self.net(x)


def test():
	features = torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.5])
	weights = nn.parameter.Parameter(torch.rand(5))
	



test()