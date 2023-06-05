import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from trainer.base_trainer import BaseTrainer
from utils.metrics import Metrics

from typing import Dict

class LacTrainer(BaseTrainer):
	def __init__(
		self: "LacTrainer", 
		model: nn.Module, 
		config: Dict, 
		device: torch.device,
		data_loader: DataLoader, 
		log_path: str
	) -> None:
		super().__init__(model, config, None, None, log_path)
		self.config = config
		self.device = device
		self.data_loader = data_loader

	def _train_epoch(self: "LacTrainer", epoch: int):
		self.model.train()
		for x, _ in tqdm(self.data_loader):
			x = x.to(self.device)

			target_hidden_activities, target_output = self.model(x)
			prediction_hidden_activities, _ = self.model(target_output)

			hidden_loss = torch.sum((target_hidden_activities.detach() - prediction_hidden_activities)**2)
			output_loss = torch.sum((target_output - x)**2)

			# hidden layer update
			self.model.hidden_optim.zero_grad()
			hidden_loss.backward(retain_graph=True)
			self.model.hidden_optim.step()

			# output layer update
			self.model.output_optim.zero_grad()
			output_loss.backward()
			self.model.output_optim.step()

			with torch.no_grad():
				self.metrics.add(epoch, len(x), [output_loss.cpu().item(), hidden_loss.cpu().item()])

		#dot = make_dot(output_loss)
		#dot.format = 'png'
		#dot.render('./image/computation_graph')

