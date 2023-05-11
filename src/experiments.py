import torch.nn as nn

from tester.base_tester import BaseTester
from tester.tester import Tester
from trainer.trainer import Trainer
from trainer.base_trainer import BaseTrainer
from utils.get_dataloader import get_dataloader
from utils.get_result_plotting import get_result_plotting
from model.autoencoder import AutoEncoder

from typing import Dict

class Experiments:
	def __init__(
		self: "Experiments",
		config: Dict,
	) -> None:
		self.config = config
		data_config = config['data']
		model_config = config['model']

		self.train_data_loader,  self.test_data_loader = get_dataloader(data_config['dataset'], 
			       train_batch_size=data_config['train_batch_size'], test_batch_size=data_config['test_batch_size'])
		self.result_plotting = get_result_plotting(data_config['dataset'])

		self.in_features = model_config['in_out_features']
		self.hidden_sizes = model_config['hidden_sizes']
		self.device = model_config['device']
	
	def default_auto_encoder(self: "Experiments"):
		autoencoder = AutoEncoder(self.in_features, self.hidden_sizes, self.device)
		trainer = Trainer(autoencoder, self.config, self.device, self.train_data_loader)
		tester = Tester(autoencoder, self.config, self.device, self.test_data_loader, self.result_plotting)

		trainer.train()
		tester.test()


