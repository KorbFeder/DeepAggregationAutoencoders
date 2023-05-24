from typing import Dict
import torch.nn as nn
from tqdm import tqdm
from trainer.base_trainer import BaseTrainer
from torch.utils.data import DataLoader
from model.edge_selection_autoencoder import EdgeSelctionAutoencoder

class EdgeSelectionTrainer(BaseTrainer):
	def __init__(
		self: "EdgeSelectionTrainer", 
		model: EdgeSelctionAutoencoder, 
		config: Dict, 
		data_loader: DataLoader,
		error = nn.MSELoss(),
	) -> None:
		super().__init__(model, config, None, None)
		self.error = error
		self.data_loader = data_loader
		self.model = model

	def _train_epoch(self: "EdgeSelectionTrainer", epoch: int) -> None:
		for batch_features, _ in tqdm(self.data_loader):
			prediction = self.model.train(batch_features)
			train_loss = self.error(prediction, batch_features)

			self.metrics.add(epoch, len(batch_features), [train_loss.item()])