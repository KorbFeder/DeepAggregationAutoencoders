import torch
import logging
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data import DataLoader

from trainer.base_trainer import BaseTrainer
from archived_models.diff_edge_node_ae import DiffEdgeNodeAutoencoder
from archived_models.diff_edge_node_ae import TrainMode

from typing import Dict

class DiffEdgeTrainer(BaseTrainer):
	def __init__(
		self: "DiffEdgeTrainer", 
		model: DiffEdgeNodeAutoencoder, 
		config: Dict,
		device: torch.device,
		data_loader: DataLoader, 
		log_path: str,
		train_mode: TrainMode = TrainMode.train_nodes,
		error = nn.MSELoss(),
	) -> None:
		super().__init__(model, config, None, None, log_path)
		self.error = error
		self.data_loader = data_loader
		self.device = device
		self.config = config
		self.train_mode = train_mode

	def _train_epoch(self: "DiffEdgeTrainer", epoch: int) -> None:
		optim = Adam(self.model.parameters(), lr=1e-3)
		flatten = nn.Flatten()

		self.model.train()

		print(f'Train mode: {self.train_mode.name}')
		for batch_features, _ in tqdm(self.data_loader):

			batch_features = flatten(batch_features).to(self.device)

			optim.zero_grad()

			output = self.model(batch_features, train_mode = self.train_mode)

			train_loss = self.error(output, batch_features)

			train_loss.backward()
			optim.step()

			# save data for plotting
			with torch.no_grad():
				self.metrics.add(epoch, len(batch_features), [train_loss.item()])
