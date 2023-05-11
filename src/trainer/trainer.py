import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data import DataLoader

from trainer.base_trainer import BaseTrainer

from typing import Dict

class Trainer(BaseTrainer):
	def __init__(
		self: "Trainer", 
		model: nn.Module, 
		config: Dict,
		device: torch.device,
		data_loader: DataLoader, 
		error = nn.MSELoss(),
	) -> None:
		super().__init__(model, config, None, None)
		self.error = error
		self.data_loader = data_loader
		self.device = device
		self.config = config

	def _train_epoch(self: "Trainer", epoch: int) -> None:
		optim = Adam(self.model.parameters(), lr=1e-3)
		flatten = nn.Flatten()

		self.model.train()
		for batch_features, _ in tqdm(self.data_loader):

			batch_features = flatten(batch_features).to(self.device)

			optim.zero_grad()

			output = self.model(batch_features)

			train_loss = self.error(output, batch_features)

			train_loss.backward()
			optim.step()

			# save data for plotting
			with torch.no_grad():
				self.metrics.add(epoch, len(batch_features), [train_loss.item()])


	