import torch
import os
import yaml
import argparse
import logging
from globals.folder_names import LOG_FOLDER
from model.diff_edge_node_ae import TrainMode, T_Conorm, T_Norm, EdgeType

from experiments import Experiments

def parse_args():
	parser = argparse.ArgumentParser(description="Training")

	parser.add_argument('--config', '-c', default='wine', type=str, 
		help='Configuration file. Use name without extension and without file path')
	
	return parser.parse_args()

def get_config_path(args) -> str:
	config_path = './config'
	path = os.path.join(config_path, args.config + '.yaml')
	return path


	
if __name__ == "__main__":
	args = parse_args()
	file = open(get_config_path(args), 'r')

	config = yaml.load(file, Loader=yaml.Loader)

	if config['model']['device'] == 'cuda':
		device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	elif config['model']['device'] == 'cpu':
		device = torch.device('cpu')

	experiments = Experiments(config)
	experiments.default_autoencoder()
	#experiments.deep_aggr_autoenc()
	#experiments.ddlg_autoencoder()
	#experiments.edge_autoencoder()
	#experiments.edge_powerset_autoencoder()
	#experiments.edge_selection_autoencoder()
	#experiments.diff_edge_autoencoder()
	#experiments.diff_edge_node_ae()
	#experiments.owa_autoencoder()
	#experiments.diff_sample_ae()
	#experiments.edge_counting()
	#args = [
	#	[[TrainMode.train_edges], [T_Norm.min, T_Conorm.max], [EdgeType.no_edge, EdgeType.normal_edge], False],
##		[[TrainMode.train_edges], [T_Norm.min, T_Conorm.max], [EdgeType.no_edge, EdgeType.normal_edge], True],
##		[[TrainMode.train_nodes, TrainMode.train_edges], [T_Norm.min, T_Conorm.max], [EdgeType.no_edge, EdgeType.normal_edge], False],
##		[[TrainMode.train_nodes, TrainMode.train_edges], [T_Norm.min, T_Conorm.max], [EdgeType.no_edge, EdgeType.normal_edge], True]
	#]
	#experiments.compare_experiments([experiments.default_autoencoder ,experiments.edge_counting])