import torch

from tester.tester import Tester
from trainer.trainer import Trainer
from trainer.ddlg_trainier import DdlgTrainer
from trainer.deep_aggr_trainer import DeepAggregateTrainer
from trainer.edge_selection_trainer import EdgeSelectionTrainer
from utils.get_dataloader import get_dataloader
from utils.get_result_plotting import get_result_plotting
from model.autoencoder import AutoEncoder
from model.deep_aggr_autoencoder import DeepAggregateAutoEncoder
from model.ddlg_autoencoder import DdlgAutoencoder
from model.edge_autoencoder import EdgeAutoencoder
from model.edge_powerset_autoencoder import EdgePowersetAutoencoder
from model.edge_selection_autoencoder import EdgeSelctionAutoencoder
from model.diff_edge_autoencoder import DiffEdgeAutoencoder

from logger.ddlg_neurons import ddlg_neurons
from logger.diff_edges_visualized import diff_edges_visualized


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
	
	def default_autoencoder(self: "Experiments") -> None:
		autoencoder = AutoEncoder(self.in_features, self.hidden_sizes, self.device)
		trainer = Trainer(autoencoder, self.config, self.device, self.train_data_loader)
		tester = Tester(autoencoder, self.config, self.device, self.test_data_loader, self.result_plotting)

		trainer.train()
		tester.test()
	
	def deep_aggr_autoenc(self: "Experiments") -> None:
		deep_aggr_ae = DeepAggregateAutoEncoder(self.in_features, self.hidden_sizes, [8, 8, 8, 8])
		trainer = DeepAggregateTrainer(deep_aggr_ae, self.config, self.train_data_loader)
		tester = Tester(deep_aggr_ae, self.config, torch.device('cpu'), self.test_data_loader, self.result_plotting)

		trainer.train()
		tester.test()

	def ddlg_autoencoder(self: "Experiments") -> None:
		ddlg_ae = DdlgAutoencoder(self.in_features, self.hidden_sizes, 4, self.device)
		trainer = DdlgTrainer(ddlg_ae, self.config, self.device, self.train_data_loader)
		tester = Tester(ddlg_ae, self.config, self.device, self.test_data_loader, self.result_plotting)

		trainer.train()
		ddlg_neurons(ddlg_ae)
		tester.test()

	def edge_autoencoder(self: "Experiments") -> None:
		edge_ae = EdgeAutoencoder(self.in_features, self.hidden_sizes, self.device)
		trainer = Trainer(edge_ae, self.config, self.device, self.train_data_loader)
		tester = Tester(edge_ae, self.config, self.device, self.test_data_loader, self.result_plotting)

		trainer.train()
		tester.test()

	def edge_powerset_autoencoder(self: "Experiments") -> None:
		edge_ae = EdgePowersetAutoencoder(self.in_features, self.hidden_sizes, self.device)
		trainer = Trainer(edge_ae, self.config, self.device, self.train_data_loader)
		tester = Tester(edge_ae, self.config, self.device, self.test_data_loader, self.result_plotting)

		trainer.train()
		tester.test()

	def edge_selection_autoencoder(self: "Experiments") -> None: 
		edge_ae = EdgeSelctionAutoencoder(self.in_features, self.hidden_sizes, seed=0)
		trainer = EdgeSelectionTrainer(edge_ae, self.config, self.train_data_loader)
		tester = Tester(edge_ae, self.config, torch.device('cpu'), self.test_data_loader, self.result_plotting)

		trainer.train()
		tester.test()

	def diff_edge_autoencoder(self: "Experiments") -> None:
		edge_ae = DiffEdgeAutoencoder(self.in_features, self.hidden_sizes, self.device)
		trainer = DdlgTrainer(edge_ae, self.config, self.device, self.train_data_loader)
		tester = Tester(edge_ae, self.config, self.device, self.test_data_loader, self.result_plotting)

		trainer.train()
		diff_edges_visualized(edge_ae)
		tester.test()




