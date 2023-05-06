import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from trainer.base_trainer import BaseTrainer

from typing import Dict

class DeepAggregateTrainer(BaseTrainer):
	def __init__(
		self: "DeepAggregateTrainer",
		model: nn.Module,
		data_loader: DataLoader,
		config: Dict
	) -> None:
		super().__init__(model, None, None, config)
		self.model = model
		self.data_loader = data_loader

	def _train_epoch(self: "DeepAggregateTrainer", epoch: int) -> None: 
		for x, _ in tqdm(self.data_loader):
			output, target_activation = self.model(x, True)
			_, prediction_activation = self.model(output, True)

			loss = ((target_activation - prediction_activation) ** 2)
			print(loss)
