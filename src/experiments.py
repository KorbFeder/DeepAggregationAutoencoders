import torch
import os

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
from logger.plot_loss import plot_loss
from logger.diff_edges_visualized import diff_edges_visualized
from logger.print_avg_loss import print_avg_loss
from utils.metrics import Metrics
from utils.create_experiment_log_dir import create_experiment_log_dir
from globals.folder_names import LOG_FOLDER, IMAGE_FOLDER

from typing import Dict, List, Tuple, Callable, Optional

class Experiments:
	def __init__(
		self: "Experiments",
		config: Dict,
	) -> Tuple[Metrics, Metrics]:
		self.config = config
		data_config = config['data']
		model_config = config['model']

		self.train_data_loader,  self.test_data_loader = get_dataloader(data_config['dataset'], 
			       train_batch_size=data_config['train_batch_size'], test_batch_size=data_config['test_batch_size'])
		self.result_plotting = get_result_plotting(data_config['dataset'])

		self.in_features = model_config['in_out_features']
		self.hidden_sizes = model_config['hidden_sizes']
		self.device = model_config['device']

		self.experiment_dir: str = create_experiment_log_dir(config)
		self.log_experiment_dir: str = os.path.join(self.experiment_dir, LOG_FOLDER)
		self.image_experiment_dir: str = os.path.join(self.experiment_dir, IMAGE_FOLDER)
	
	def compare_experiments(self: "Experiments", experiments: List[Callable[[], Tuple[Metrics, Metrics]]]) -> None:
		all_train_metrics = []
		label_name = []
		all_test_metrics = []
		i = 0
		name = self.config['path']['experiment_name']
		for experiment in experiments:
			self.config['path']['experiment_name'] = name + str(i)
			train_metrics, test_metrics = experiment()
			label_name.append(experiment.__name__)
			all_train_metrics.append(train_metrics)
			all_test_metrics.append(test_metrics)
			i += 1
		self.config['path']['experiment_name'] = name

		self._compare_experiments_plot(all_train_metrics, label_name)

		for train_metrics, test_metrics, label in zip(all_train_metrics, all_test_metrics, label_name):
			print_avg_loss(train_metrics, test_metrics, label)
	
	def _compare_experiments_plot(self: "Experiments", result_metrics: List[Metrics], label_name: Optional[List[str]] = None):
		path_config = self.config['path']
		per_sample_losses = []
		episodic_losses = []
		for metric in result_metrics:
			per_sample_losses.append(metric.per_sample_loss[0])
			episodic_losses.append(metric.episodic_loss[0])

		name = path_config['experiment_name']
		plot_loss(per_sample_losses, self.image_experiment_dir, f'comparison-per-sample-loss-{name}', legend=label_name)
		plot_loss(episodic_losses, self.image_experiment_dir, f'comparison-episodic-{name}', legend=label_name)
	
	def default_autoencoder(self: "Experiments") -> Tuple[Metrics, Metrics]:
		autoencoder = AutoEncoder(self.in_features, self.hidden_sizes, self.device)
		trainer = Trainer(autoencoder, self.config, self.device, self.train_data_loader, self.experiment_dir)
		tester = Tester(autoencoder, self.config, self.device, self.test_data_loader, self.experiment_dir, self.result_plotting)

		return trainer.train(), tester.test()
	
	def deep_aggr_autoenc(self: "Experiments") -> Tuple[Metrics, Metrics]:
		deep_aggr_ae = DeepAggregateAutoEncoder(self.in_features, self.hidden_sizes, [8, 8, 8, 8])
		trainer = DeepAggregateTrainer(deep_aggr_ae, self.config, self.train_data_loader, self.experiment_dir)
		tester = Tester(deep_aggr_ae, self.config, torch.device('cpu'), self.test_data_loader, self.experiment_dir, self.result_plotting)

		return trainer.train(), tester.test()

	def ddlg_autoencoder(self: "Experiments") -> Tuple[Metrics, Metrics]:
		ddlg_ae = DdlgAutoencoder(self.in_features, self.hidden_sizes, 4, self.device)
		trainer = DdlgTrainer(ddlg_ae, self.config, self.device, self.train_data_loader, self.experiment_dir)
		tester = Tester(ddlg_ae, self.config, self.device, self.test_data_loader, self.experiment_dir, self.result_plotting)

		train_metric = trainer.train()
		ddlg_neurons(ddlg_ae)
		return train_metric, tester.test()

	def edge_autoencoder(self: "Experiments") -> Tuple[Metrics, Metrics]:
		edge_ae = EdgeAutoencoder(self.in_features, self.hidden_sizes, self.device)
		trainer = Trainer(edge_ae, self.config, self.device, self.train_data_loader, self.experiment_dir)
		tester = Tester(edge_ae, self.config, self.device, self.test_data_loader, self.experiment_dir, self.result_plotting)

		return trainer.train(), tester.test()

	def edge_powerset_autoencoder(self: "Experiments") -> Tuple[Metrics, Metrics]:
		edge_ae = EdgePowersetAutoencoder(self.in_features, self.hidden_sizes, self.device)
		trainer = Trainer(edge_ae, self.config, self.device, self.train_data_loader, self.experiment_dir)
		tester = Tester(edge_ae, self.config, self.device, self.test_data_loader, self.experiment_dir, self.result_plotting)

		return trainer.train(), tester.test()

	def edge_selection_autoencoder(self: "Experiments") -> Tuple[Metrics, Metrics]: 
		edge_ae = EdgeSelctionAutoencoder(self.in_features, self.hidden_sizes, seed=0)
		trainer = EdgeSelectionTrainer(edge_ae, self.config, self.train_data_loader, self.experiment_dir)
		tester = Tester(edge_ae, self.config, torch.device('cpu'), self.test_data_loader, self.experiment_dir, self.result_plotting)

		return trainer.train(), tester.test()

	def diff_edge_autoencoder(self: "Experiments") -> Tuple[Metrics, Metrics]:
		edge_ae = DiffEdgeAutoencoder(self.in_features, self.hidden_sizes, self.device)
		trainer = DdlgTrainer(edge_ae, self.config, self.device, self.train_data_loader, self.experiment_dir)
		tester = Tester(edge_ae, self.config, self.device, self.test_data_loader, self.experiment_dir, self.result_plotting)

		train_metrcis = trainer.train()
		diff_edges_visualized(edge_ae)
		return train_metrcis, tester.test()




