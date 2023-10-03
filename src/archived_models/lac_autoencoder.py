import torch
import torch.nn as nn
from torch.optim import Adam
from torchviz import make_dot

from typing import List, Tuple, TypedDict, Optional

class LayerComponents(TypedDict):
	layer_type: nn.Module
	activation: nn.Module
	normalization: Optional[nn.Module]

# implementation of the local activity contrast (LAC) algorithm implemented in:
# https://doi.org/10.3390/electronics12010147   

class LacAutoEncoder(nn.Module):
	def __init__(
		self: "LacAutoEncoder",
		in_features: int,
		hidden_sizes: List[int],
		device: torch.device,
		activation: nn.Module = nn.ReLU(),
		out_activation: nn.Module = nn.Sigmoid()
	) -> None:
		super().__init__()

		self.device = device
		layer_sizes = [in_features, *hidden_sizes]
		self.layers: List[LayerComponents] = []
		self.num_hidden_neurons = sum(hidden_sizes)

		for i in range(len(layer_sizes)-1):
			self.layers += [{
				'layer_type': nn.Linear(layer_sizes[i], layer_sizes[i + 1]).to(self.device),
				'activation': activation.to(self.device),
				'normalization': nn.BatchNorm1d(layer_sizes[i + 1]).to(self.device)
			}]

		self.out_layer: LayerComponents = {
			'layer_type': nn.Linear(layer_sizes[-1], in_features).to(self.device), 
			'activation': out_activation.to(self.device), 
			'normalization': None
		}

		h_params = sum([list(layer['layer_type'].parameters()) + list(layer['normalization'].parameters()) for layer in self.layers[:-1]], [])
		o_params = self.out_layer['layer_type'].parameters()

		# parameter from paper which where used for mnist 
		self.hidden_optim = Adam(h_params, lr=0.001, weight_decay=0.001)
		self.output_optim = Adam(o_params, lr=0.01, weight_decay=0.001)

	def forward(self: "LacAutoEncoder", x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		hidden_activations = torch.Tensor(x.shape[0], self.num_hidden_neurons).to(self.device)
		i = 0

		# forward pass through the network, but save the activation of every neuron
		A = x
		for layer in self.layers: 
			WA = (layer['layer_type'](A.detach()))
			S = layer['normalization'](WA)
			A = layer['activation'](S)

			new_i = S.shape[1] + i
			hidden_activations[:,i:new_i] = S
			i = new_i

		# the last layers output
		output = self.out_layer['activation'](self.out_layer['layer_type'](A.detach()))
		#dot = make_dot(output, params=dict(self.out_layer['layer_type'].named_parameters()))
		#dot.format = 'png'
		#dot.render('./image/computation_graph')

		return hidden_activations, output