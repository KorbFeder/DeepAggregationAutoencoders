import torch.nn as nn
from abc import abstractmethod

class BaseTrainer:
	def __init__(
		self: "BaseTrainer",
		model: nn.Module,
		criterion: nn.Module,
		optimizer: nn.Module,
		config
	) -> None:
		self.config = config
		self.model = model
		self.criterion = criterion
		self.optimizer = optimizer

		trainer_config = config['trainer']	
		self.epochs = trainer_config['epochs']
	
		self.start_epoch = 1

	@abstractmethod
	def _train_epoch(self: "BaseTrainer", epoch: int):
		raise NotImplementedError

	def train(self: "BaseTrainer"):
		for epoch in range(self.start_epoch, self.epochs + 1):
			print(f"training {epoch}/{self.epochs} ...")
			result = self._train_epoch(epoch)

