import torch
import statistics
from statistics import mean
import os
import numpy as np
import math

from tester.tester import Tester
from trainer.trainer import Trainer
from trainer.ddlg_trainier import DdlgTrainer
from trainer.deep_aggr_trainer import DeepAggregateTrainer
from trainer.edge_selection_trainer import EdgeSelectionTrainer
from trainer.diff_edge_node_trainer import DiffEdgeTrainer
from utils.get_dataloader import get_dataloader
from utils.get_result_plotting import get_result_plotting
from utils.set_random_seed import set_random_seed
from archived_models.deep_aggr_autoencoder import DeepAggregateAutoEncoder
from archived_models.ddlg_autoencoder import DdlgAutoencoder
from archived_models.edge_autoencoder import EdgeAutoencoder
from archived_models.edge_powerset_autoencoder import EdgePowersetAutoencoder
from archived_models.edge_selection_autoencoder import EdgeSelctionAutoencoder
from archived_models.diff_edge_autoencoder import DiffEdgeAutoencoder
from archived_models.diff_edge_node_ae import DiffEdgeNodeAutoencoder, TrainMode, T_Conorm, T_Norm, EdgeType
from archived_models.owa_autoencoder import OwaAutoencoder
from archived_models.diff_sample_ae import DiffSampleAutoencoder
from archived_models.node_counting import NodeCountingAutoencoder
from archived_models.forward_forward_counting import ForwardForwardCoutingAutoencoder
from archived_models.forward_forward_counting_v2 import FFEdgeCountingAutoencoder
from archived_models.forward_forward_counting_v3 import FFEdgeCountingAutoencoder3
from archived_models.forward_forward_counting_v4 import FFEdgeCountingAutoencoder4
from archived_models.forward_forward_node_edge_counting  import ForwardForwardNodeEdgeCoutingAutoencoder
from archived_models.node_edge_counting import EdgeNodeCountingAutoencoder

from model.autoencoder import AutoEncoder
from model.edge_counting import EdgeCountingAutoencoder
from model.daa import DAAutoencoder, NodeInitailzation


from logger.ddlg_neurons import ddlg_neurons
from logger.plot_loss import plot_loss
from logger.diff_edges_visualized import diff_edges_visualized
from logger.print_avg_loss import print_avg_loss
from logger.edge_counts import print_edge_counts
from logger.node_counts import print_node_counts
from logger.print_operators import print_operators
from logger.stddev_plot import stddev_plot
from logger.print_logic_formula import print_logic_formula, print_hidden_logic_formula, print_out_logic_formula
from logger.plot_net import plot_net
from logger.repeated_exp_to_csv import repeated_exp_to_csv
from utils.metrics import Metrics
from utils.create_experiment_log_dir import create_experiment_log_dir
from utils.configure_logger import configure_logger
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
		self.seed = model_config['seed']
		set_random_seed(model_config['seed'])
		self.autoencoder = None

		self.train_data_loader,  self.test_data_loader = get_dataloader(data_config['dataset'], 
			       train_batch_size=data_config['train_batch_size'], test_batch_size=data_config['test_batch_size'])
		self.result_plotting = get_result_plotting(data_config['dataset'])

		self.in_features = model_config['in_out_features']
		self.hidden_sizes = model_config['hidden_sizes']
		self.device = model_config['device']

		self.experiment_dir: str = create_experiment_log_dir(config)
		self.log_experiment_dir: str = os.path.join(self.experiment_dir, LOG_FOLDER)
		self.image_experiment_dir: str = os.path.join(self.experiment_dir, IMAGE_FOLDER)
		configure_logger(config['path']['logger_level'], os.path.join(self.log_experiment_dir, 'logging.log'))
	

	def repeat_experiment(self: "Experiments", experiment: Callable[[], Tuple[Metrics, Metrics]], iterations: int, 
		arguments = None, plot_std_bands: bool = True):
		all_train_metrics = []
		all_test_metrics = []
		name = self.config['path']['experiment_name']

		# set seed to None, to case randomness 
		prev_seed = self.config['model']['seed']
		self.config['model']['seed'] = None
		self.seed = None
		label_name = []
		
		for i in range(iterations):
			self.config['path']['experiment_name'] = name + str(i)
			if arguments:
				if arguments == None:
					train_metrics, test_metrics = experiment()
				else:
					train_metrics, test_metrics = experiment(*(arguments))
			else:
				train_metrics, test_metrics = experiment()
			label_name.append(experiment.__name__)
			all_train_metrics.append(train_metrics)
			all_test_metrics.append(test_metrics)


		# set changed config back
		self.config['path']['experiment_name'] = name
		self.config['model']['seed'] = prev_seed
		self.seed = prev_seed

		self.avg_loss(all_test_metrics, os.path.join(self.log_experiment_dir, 'avg_test_loss.txt'))
		self.avg_loss(all_train_metrics, os.path.join(self.log_experiment_dir, 'avg_train_loss.txt'))

		self._std_plot(all_train_metrics, label_name, use_bands=plot_std_bands)
	
	def compare_experiments(self: "Experiments", experiments: List[Callable[[], Tuple[Metrics, Metrics]]], arguments = None) -> None:
		all_train_metrics = []
		label_name = []
		all_test_metrics = []
		name = self.config['path']['experiment_name']
	
		for i, experiment in enumerate(experiments):
			self.config['path']['experiment_name'] = name + str(i)
			if arguments:
				if arguments[i] == None:
					train_metrics, test_metrics = experiment()
				else:
					train_metrics, test_metrics = experiment(*(arguments[i]))
			else:
				train_metrics, test_metrics = experiment()
			label_name.append(experiment.__name__)
			all_train_metrics.append(train_metrics)
			all_test_metrics.append(test_metrics)
		self.config['path']['experiment_name'] = name

		self._compare_experiments_plot(all_train_metrics, label_name)

		for train_metrics, test_metrics, label in zip(all_train_metrics, all_test_metrics, label_name):
			print_avg_loss(train_metrics, test_metrics, label)
	
	def repat_multiple_experiments(self: "Experiments", experiments: List[Callable[[], Tuple[Metrics, Metrics]]], iterations: int, 
		arguments = None, plot_std_bands: bool = True, legend=None):
		all_train_metrics = []
		all_test_metrics = []
		name = self.config['path']['experiment_name']

		# set seed to None, to case randomness 
		prev_seed = self.config['model']['seed']
		self.config['model']['seed'] = None
		self.seed = None

		label_name = []
		autoencoder = None

		for u, experiment in enumerate(experiments):
			exp_train_metrics = []
			exp_test_metrics = []
			minimum = math.inf
			autoencoder = None

			for i in range(iterations):
				self.config['path']['experiment_name'] = name + str(i)
				if arguments[u]:
					if arguments[u] == None:
						train_metrics, test_metrics = experiment()
					else:
						train_metrics, test_metrics = experiment(*(arguments[u]))
				else:
					train_metrics, test_metrics = experiment()
				label_name.append(experiment.__name__)

				exp_train_metrics.append(train_metrics)
				exp_test_metrics.append(test_metrics)
				
				avg_test_loss = mean(test_metrics.per_sample_loss[0])
				if mean(test_metrics.per_sample_loss[0]) < minimum:
					minimum = avg_test_loss
					autoencoder = self.autoencoder

			if autoencoder != None: 
				print_hidden_logic_formula(self.autoencoder, path=os.path.join(self.log_experiment_dir))
				print_out_logic_formula(self.autoencoder, path=os.path.join(self.log_experiment_dir))

			if legend:
				self.avg_loss(exp_test_metrics, os.path.join(self.log_experiment_dir, f'avg_test_loss_{legend[u]}.txt'))
				self.avg_loss(exp_train_metrics, os.path.join(self.log_experiment_dir, f'avg_train_loss_{legend[u]}.txt'))
			else:
				self.avg_loss(exp_test_metrics, os.path.join(self.log_experiment_dir, f'avg_test_loss_{u}.txt'))
				self.avg_loss(exp_train_metrics, os.path.join(self.log_experiment_dir, f'avg_train_loss_{u}.txt'))
	

			all_train_metrics.append(exp_train_metrics)
			all_test_metrics.append(exp_test_metrics)

		#repeated_exp_to_csv(all_train_metrics, all_test_metrics, self.log_experiment_dir)

		# set changed config back
		self.config['path']['experiment_name'] = name
		self.config['model']['seed'] = prev_seed
		self.seed = prev_seed

		self._std_plot(all_train_metrics, legend, use_bands=False)
		if plot_std_bands:
			self._std_plot(all_train_metrics, legend, use_bands=True)

	def _std_plot(self: "Experiments", _result_metrics: List[List[Metrics]], label_names: List[str], splits: int = 50, use_bands=True):
		path_config = self.config['path']
		per_sample_losses = []
		mean_losses = []
		episodic_losses = []
		sample_nr = []
		for result_metrics in _result_metrics:
			exp_mean_losses = []
			exp_sample_losses = []
			exp_episodic_losses = []
			for metric in result_metrics:
				mean_loss = []
				exp_sample_losses.append(metric.per_sample_loss[0])
				xs = metric.per_sample_loss[0]
				per_mean_samples = int(len(xs) / splits)
				chunks = list(xs[i:i+per_mean_samples] for i in range(0, len(xs)-1, per_mean_samples))
				sample_nr = list(str(i) for i in range(0, len(xs), per_mean_samples))
				for chunk in chunks:
					mean_loss.append(statistics.mean(chunk))
				exp_mean_losses.append(mean_loss)
				exp_episodic_losses.append(metric.episodic_loss[0])

			name = path_config['experiment_name']

			mean_losses.append(exp_mean_losses)
			per_sample_losses.append(exp_sample_losses)
			episodic_losses.append(exp_episodic_losses)

		stddev_plot(mean_losses, self.image_experiment_dir, f'mean-sample-loss-{name}', x_label='samples', x_ticks=sample_nr, use_bands=use_bands, legend=label_names)
		stddev_plot(per_sample_losses, self.image_experiment_dir, f'sample-loss-{name}', x_label='samples', use_bands=use_bands, legend=label_names)
		stddev_plot(episodic_losses, self.image_experiment_dir, f'spisodic-{name}', use_bands=use_bands, legend=label_names)
	
	def avg_loss(self: "Experiments", metrics: List[Metrics], save_file = None):
		losses = []
		times = []
		for metric in metrics:
			losses.append([mean(loss) for loss in metric.per_sample_loss][0])
			times.append(metric.curr_time_elapsed)
		if save_file:
			f = open(save_file, 'w')
			print(f'mean loss: {mean(losses)}\nbest loss: {min(losses)}\nworst loss: {max(losses)}\
			\nmin time: {min(times)}\nmax time: {max(times)}\nmean time: {mean(times)}', file=f)
		print(f'mean loss: {mean(losses)}\nbest loss: {min(losses)}\nworst loss: {max(losses)}')

	
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
	
	def default_autoencoder(self: "Experiments", use_plotting = False) -> Tuple[Metrics, Metrics]:
		autoencoder = AutoEncoder(self.in_features, self.hidden_sizes, self.device)
		trainer = Trainer(autoencoder, self.config, self.device, self.train_data_loader, self.experiment_dir)
		tester = Tester(autoencoder, self.config, self.device, self.test_data_loader, self.experiment_dir, self.result_plotting)

		if use_plotting:
			train = trainer.train()
			test = tester.test()
		else:
			train = trainer.train(save_loss_graph=False, save_train_csv_results=False)
			test = tester.test(save_csv=False, save_test_outputs=False)
	
		print_avg_loss(train, test, 'default autoencoder')

		return train, test
	
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

	def owa_autoencoder(self: "Experiments")-> Tuple[Metrics, Metrics]:
		ae = OwaAutoencoder(self.in_features, self.hidden_sizes, self.device)
		trainer = Trainer(ae, self.config, self.device, self.train_data_loader, self.experiment_dir)
		tester = Tester(ae, self.config, self.device, self.test_data_loader, self.experiment_dir, self.result_plotting)

		train_metrcis = trainer.train()
		return train_metrcis, tester.test()

	def diff_edge_node_ae(self: "Experiments", train_modes = [TrainMode.train_edges], operators = [T_Norm.min, T_Conorm.max], 
		       edge_types = [EdgeType.no_edge, EdgeType.normal_edge], use_weights = False) -> Tuple[Metrics, Metrics]:

		edge_ae = DiffEdgeNodeAutoencoder(self.in_features, self.hidden_sizes, self.device, operators, edge_types, use_weights)
		tester = Tester(edge_ae, self.config, self.device, self.test_data_loader, self.experiment_dir, self.result_plotting)

		name = self.config['path']['experiment_name']
		all_train_metrics = []
		all_test_metrics = []
		for train_mode in train_modes:
			self.config['path']['experiment_name'] = name + "-" + train_mode.name
			trainer = DiffEdgeTrainer(edge_ae, self.config, self.device, self.train_data_loader, self.experiment_dir, train_mode)
			train_metrics = trainer.train()
			test_metrics = tester.test()
			all_train_metrics.append(train_metrics)
			all_test_metrics.append(test_metrics)

		self.config['path']['experiment_name'] = name

		diff_edges_visualized(edge_ae)

		for train, test in zip(all_train_metrics, all_test_metrics):
			print_avg_loss(train, test, 'diff-edge-node-ae')
		return train_metrics, test_metrics
	
	def diff_sample_ae(self: "Experiments"):
		autoencoder = DiffSampleAutoencoder(self.in_features, self.hidden_sizes, self.device)
		trainer = Trainer(autoencoder, self.config, self.device, self.train_data_loader, self.experiment_dir)
		tester = Tester(autoencoder, self.config, self.device, self.test_data_loader, self.experiment_dir, self.result_plotting)

		return trainer.train(), tester.test()

	def edge_counting(self: "Experiments", count_step_size = 0.1, use_plotting = True):
		self.autoencoder = EdgeCountingAutoencoder(self.in_features, self.hidden_sizes, self.device, seed=self.seed, count_step_size=count_step_size)
		trainer = EdgeSelectionTrainer(self.autoencoder, self.config, self.train_data_loader, self.experiment_dir, self.device)
		tester = Tester(self.autoencoder, self.config, self.device, self.test_data_loader, self.experiment_dir, self.result_plotting)

		if use_plotting:
			train = trainer.train()
			#print_edge_counts(autoencoder)
			#print_logic_formula(autoencoder)

			print_hidden_logic_formula(self.autoencoder, path=os.path.join(self.log_experiment_dir))
			print_out_logic_formula(self.autoencoder, path=os.path.join(self.log_experiment_dir))

			plot_net(self.autoencoder, os.path.join(self.image_experiment_dir, 'net_plot'))
			test = tester.test()
		else:
			train = trainer.train(save_loss_graph=False, save_train_csv_results=False)
			test = tester.test(save_csv=False, save_test_outputs=False)

		return train, test
	
	def daa(self: "Experiments", count_step_size = 0.2, use_plotting = True, node_init: NodeInitailzation = NodeInitailzation.LAYER_WISE):
		autoencoder = DAAutoencoder(self.in_features, self.hidden_sizes, self.device, seed=self.seed, count_step_size=count_step_size, node_initialization=node_init)
		trainer = EdgeSelectionTrainer(autoencoder, self.config, self.train_data_loader, self.experiment_dir, self.device)
		tester = Tester(autoencoder, self.config, self.device, self.test_data_loader, self.experiment_dir, self.result_plotting)

		train = trainer.train(save_loss_graph=False, save_train_csv_results=False)
		test = tester.test(save_csv=False, save_test_outputs=False)

		return train, test

	def node_counting(self: "Experiments"):
		autoencoder = NodeCountingAutoencoder(self.in_features, self.hidden_sizes, self.device)
		trainer = EdgeSelectionTrainer(autoencoder, self.config, self.train_data_loader, self.experiment_dir, self.device)
		tester = Tester(autoencoder, self.config, self.device, self.test_data_loader, self.experiment_dir, self.result_plotting)

		train = trainer.train()
		print_node_counts(autoencoder)
		return train, tester.test()
	
	def forward_forward_counting(self: "Experiments"):
		autoencoder = ForwardForwardCoutingAutoencoder(self.in_features, self.hidden_sizes, self.device)
		trainer = EdgeSelectionTrainer(autoencoder, self.config, self.train_data_loader, self.experiment_dir, self.device)
		tester = Tester(autoencoder, self.config, self.device, self.test_data_loader, self.experiment_dir, self.result_plotting)

		train = trainer.train()
		print_operators(autoencoder)
		print_edge_counts(autoencoder)
		return train, tester.test()

	def ff_edge_counting(self: "Experiments"):
		autoencoder = FFEdgeCountingAutoencoder(self.in_features, self.hidden_sizes, self.device)
		trainer = EdgeSelectionTrainer(autoencoder, self.config, self.train_data_loader, self.experiment_dir, self.device)
		tester = Tester(autoencoder, self.config, self.device, self.test_data_loader, self.experiment_dir, self.result_plotting)

		train = trainer.train()
		print_edge_counts(autoencoder)
		return train, tester.test()
	
	def ff_edge_counting2(self: "Experiments"):
		autoencoder = FFEdgeCountingAutoencoder3(self.in_features, self.hidden_sizes, self.device)
		trainer = EdgeSelectionTrainer(autoencoder, self.config, self.train_data_loader, self.experiment_dir, self.device)
		tester = Tester(autoencoder, self.config, self.device, self.test_data_loader, self.experiment_dir, self.result_plotting)

		train = trainer.train()
		print_operators(autoencoder)
		print_edge_counts(autoencoder)
		return train, tester.test()
	


	def ff_edge_counting3(self: "Experiments"):
		autoencoder = FFEdgeCountingAutoencoder4(self.in_features, self.hidden_sizes, self.device)
		trainer = EdgeSelectionTrainer(autoencoder, self.config, self.train_data_loader, self.experiment_dir, self.device)
		tester = Tester(autoencoder, self.config, self.device, self.test_data_loader, self.experiment_dir, self.result_plotting)

		train = trainer.train()
		print_edge_counts(autoencoder)
		return train, tester.test()

	def forward_forward_node_edge_counting(self: "Experiments"):
		autoencoder = ForwardForwardNodeEdgeCoutingAutoencoder(self.in_features, self.hidden_sizes, self.device)
		trainer = EdgeSelectionTrainer(autoencoder, self.config, self.train_data_loader, self.experiment_dir, self.device)
		tester = Tester(autoencoder, self.config, self.device, self.test_data_loader, self.experiment_dir, self.result_plotting)

		train = trainer.train()
		print_edge_counts(autoencoder)
		return train, tester.test()
	
	def edge_node_counting(self: "Experiments"):
		autoencoder = EdgeNodeCountingAutoencoder(self.in_features, self.hidden_sizes, self.device)
		trainer = EdgeSelectionTrainer(autoencoder, self.config, self.train_data_loader, self.experiment_dir, self.device)
		tester = Tester(autoencoder, self.config, self.device, self.test_data_loader, self.experiment_dir, self.result_plotting)

		train = trainer.train()
		print_edge_counts(autoencoder)
		return train, tester.test()

	def repeat_edge_counting(self: "Experiments", count_step_size = 0.1, iterations=3):
		return self.repeat_experiment(self.edge_counting, iterations=iterations, arguments=[count_step_size, False])