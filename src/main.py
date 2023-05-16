import torch
import os
import yaml
import argparse

from experiments import Experiments


def parse_args():
	parser = argparse.ArgumentParser(description="Training")

	parser.add_argument('--config', '-c', default='simple', type=str, 
		help='Configuration file. Use name without extension and without file path')
	
	return parser.parse_args()

def get_config_path(args):
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
	#experiments.default_autoencoder()
	#experiments.deep_aggr_autoenc()
	experiments.ddlg_autoencoder()
