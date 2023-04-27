import torch
import os
import yaml
import argparse

from trainer.lac_trainer import LacTrainer
from model.lac_autoencoder import LacAutoEncoder
from data_loader.mnist_data_loader import MNIST_loaders
from tester.lac_tester import LacTester
from logger.plot_mnist_outputs import plot_mnist_outputs

def parse_args():
	parser = argparse.ArgumentParser(description="Training")

	parser.add_argument('--config', '-c', default='default', type=str, 
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
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	train_loader, test_loader = MNIST_loaders()
	model = LacAutoEncoder(784, [256, 64, 256], device)
	trainer = LacTrainer(model, config, device, train_loader)
	trainer.train()
	tester = LacTester(model, device, test_loader, plot_mnist_outputs)
	tester.test()


