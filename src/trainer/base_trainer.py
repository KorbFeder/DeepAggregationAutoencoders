import torch.nn as nn
from abc import abstractmethod

from typing import Dict

from utils.metrics import Metrics

class BaseTrainer:
	def __init__(
		self: "BaseTrainer",
		model: nn.Module,
		config: Dict,
		criterion: nn.Module,
		optimizer: nn.Module,
	) -> None:
		self.model = model
		self.criterion = criterion
		self.optimizer = optimizer
		self.start_epoch = 1
		self.config = config

		trainer_config = config['trainer']	
		self.epochs = trainer_config['epochs']
		path_config = config['path']
		self.csv_save_path = path_config['csv_save_path']
		self.csv_name = path_config['csv_name']
		self.plot_save_path = path_config['plot_save_path']
		self.plot_name = path_config['plot_name']
	
		self.metrics: Metrics = Metrics()

	@abstractmethod
	def _train_epoch(self: "BaseTrainer", epoch: int):
		raise NotImplementedError

	def train(self: "BaseTrainer"):
		for epoch in range(self.start_epoch, self.epochs + 1):
			print(f"training {epoch}/{self.epochs} ...")
			self._train_epoch(epoch)

			self.metrics.save(self.csv_save_path, f"train-{self.csv_name}")
			self.metrics.plot_loss(self.plot_save_path, f'train-{self.plot_name}')

