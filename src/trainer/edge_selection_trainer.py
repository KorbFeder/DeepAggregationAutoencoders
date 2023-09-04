from typing import Dict
import torch.nn as nn
import numpy as np
import torch
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
		log_path: str,
		device: torch.device,
		error = nn.MSELoss(),
	) -> None:
		super().__init__(model, config, None, None, log_path)
		self.error = error
		self.data_loader = data_loader
		self.model = model
		self.device = device

	def _train_epoch(self: "EdgeSelectionTrainer", epoch: int) -> None:
		for batch_features, _ in tqdm(self.data_loader):
			batch_features = batch_features.to(self.device)
			prediction = self.model.train(batch_features)
			train_loss = self.error(prediction, batch_features)

			if np.isnan(train_loss.item()):
				print('a')

			self.metrics.add(epoch, len(batch_features), [train_loss.item()])