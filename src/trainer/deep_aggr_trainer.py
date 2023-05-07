import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from trainer.base_trainer import BaseTrainer

from typing import Dict

from utils.metrics import Metrics

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
		self.save_path = config['csv_save_path']
		self.metrics = Metrics()

	def _train_epoch(self: "DeepAggregateTrainer", epoch: int) -> None: 
		for x, _ in tqdm(self.data_loader):
			output_loss = self.model.train(x)
			self.metrics.add(epoch, len(x), [output_loss])

		self.metrics.save(self.save_path)
		self.metrics.plot_loss('train')