import torch
import os
import yaml
import argparse

from experiments import Experiments
from model.daa import NodeInitailzation

def parse_args():
	parser = argparse.ArgumentParser(description="Training")

	parser.add_argument('--config', '-c', default='horton', type=str, 
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

	#experiments.compare_experiments([experiments.default_autoencoder, experiments.edge_counting, experiments.ff_edge_counting])
	#args = [[0.05, False], [0.2, False], [1, False]]
	#experiments.repat_multiple_experiments(
	#	[experiments.edge_counting for i in args], 
	#	5, args, 
	#	legend=[f'glass_{value[0]}' for value in args], 
	#	plot_std_bands=False
	#)
	#experiments.daa()
	experiments.edge_counting()

	#args = [[], [0.1, False]]
	#experiments.repat_multiple_experiments(
	#	[ experiments.default_autoencoder, experiments.edge_counting, ], 
	#	#[experiments.edge_counting, experiments.default_autoencoder], 
	#	20, args, 
	#	legend=['ANN-AE', 'DAA'], 
	#	plot_std_bands=True
	#)

