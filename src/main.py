import torch
import torch.nn as nn

from models.AutoEncoder import AutoEncoder
from models.DeepAggregateAutoEncoder import DeepAggregateAutoEncoder
from models.MinMaxAutoEncoder import MinMaxAutoEncoder
from models.BinaryAutoEncoder import BinaryAutoEncoder
from models.GeneticAlgTraining import GeneticAlgTraining
from Evaluation import Evaluation

#from utils.ga_test import test
from data.WineQualityDatafetcher import WineQualityDatafetcher

if __name__ == "__main__":
	wineQualityDatafetcher = WineQualityDatafetcher("./datasets/WineQuality/winequality-white.csv")
	#test()
	#fae = DeepAggregateAutoEncoder(784, [128, 64, 128], [torch.min, torch.max, torch.max, torch.min], activation=nn.ReLU)
	ae = AutoEncoder(12, [8, 4, 8])
	#mae = MinMaxAutoEncoder(784, [128, 64, 128], [torch.max, torch.max, torch.max, torch.min], activation=nn.ReLU)
	#bae = BinaryAutoEncoder(784, [128], [torch.max, torch.max])
	#ooga = GeneticAlgTraining(ae)
	#ooga_model = ooga.train()
	#evaluation = Evaluation(ooga_model)
	
	evaluation = Evaluation(ae, wineQualityDatafetcher, 30)
	#evaluation = Evaluation(fae, 1)
	##evaluation = Evaluation(mae, 80)
	evaluation.train()
	evaluation.test()


