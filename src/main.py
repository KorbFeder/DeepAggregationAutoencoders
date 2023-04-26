import torch
import torch.nn as nn

from models.AutoEncoder import AutoEncoder
from models.DeepAggregateAutoEncoder import DeepAggregateAutoEncoder
from models.MinMaxAutoEncoder import MinMaxAutoEncoder
from models.BinaryAutoEncoder import BinaryAutoEncoder
from models.GeneticAlgTraining import GeneticAlgTraining
from Evaluation import Evaluation

from data.WineQualityDatafetcher import WineQualityDatafetcher
from data.MnistDatafetcher import MnistDatafetcher
from utils.plotting import plot_mnist_outputs, plot_wine_results
from fuzzy_logic.HighMembership import HighMembership

#from models.ForwardForward import test
from models.ForwardForwardLayer import test

if __name__ == "__main__":
	test()
	membership = HighMembership()
	wineQualityDatafetcher = WineQualityDatafetcher("./datasets/WineQuality/winequality-white.csv", transform=membership.fuzzification)
	mnistDatafetcher = MnistDatafetcher()
	#test()
	#fae = DeepAggregateAutoEncoder(784, [128, 64, 128], [torch.min, torch.max, torch.max, torch.min], activation=nn.ReLU)
	#ae = AutoEncoder(wineQualityDatafetcher.num_features(), [10, 6, 4, 6, 10])
	mae = DeepAggregateAutoEncoder(wineQualityDatafetcher.num_features(), [10, 6, 4, 6, 10], [torch.max, torch.min, torch.max, torch.min, torch.max, torch.min], activation=nn.ReLU)
	#bae = BinaryAutoEncoder(784, [128], [torch.max, torch.max])
	#ooga = GeneticAlgTraining(ae)
	#ooga_model = ooga.train()
	#evaluation = Evaluation(ooga_model)
	
	evaluation = Evaluation(mae, wineQualityDatafetcher, 10, backtransformation=membership.defuzzification, plot_outputs=plot_wine_results)
	#evaluation = Evaluation(fae, 1)
	##evaluation = Evaluation(mae, 80)
	evaluation.train()
	evaluation.test()


