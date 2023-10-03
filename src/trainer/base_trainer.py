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
		log_path: str
	) -> None:
		self.model = model
		self.criterion = criterion
		self.optimizer = optimizer
		self.start_epoch = 1
		self.config = config


		trainer_config = config['trainer']	
		self.epochs = trainer_config['epochs']
		path_config = config['path']
		dataset = config['data']['dataset']
		self.experiment_name = f"{dataset}-{path_config['experiment_name']}"
	
		self.metrics: Metrics = Metrics(log_path)

	@abstractmethod
	def _train_epoch(self: "BaseTrainer", epoch: int):
		raise NotImplementedError

	def train(self: "BaseTrainer", save_loss_graph: bool = True, save_train_csv_results: bool = True) -> Metrics:
		for epoch in range(self.start_epoch, self.epochs + 1):
			print(f"training {epoch}/{self.epochs} ...")
			self._train_epoch(epoch)

			if save_train_csv_results:
				self.metrics.save(f"train-result.csv")
			if save_loss_graph:
				self.metrics.plot_loss(f'train-{self.experiment_name}')
			self.metrics.print_last()

		return self.metrics

