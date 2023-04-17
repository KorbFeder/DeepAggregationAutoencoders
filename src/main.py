import torch
import torch.nn as nn

from models.AutoEncoder import AutoEncoder
from models.DeepAggregateAutoEncoder import DeepAggregateAutoEncoder
from models.MinMaxAutoEncoder import MinMaxAutoEncoder
from models.BinaryAutoEncoder import BinaryAutoEncoder
from models.GeneticAlgTraining import GeneticAlgTraining
from Evaluation import Evaluation

if __name__ == "__main__":
	fae = DeepAggregateAutoEncoder(784, [128, 64, 128], [torch.min, torch.max, torch.max, torch.min], activation=nn.ReLU)
	ae = AutoEncoder(784, [128, 64, 128])
	mae = MinMaxAutoEncoder(784, [128, 64, 128], [torch.max, torch.max, torch.max, torch.min], activation=nn.ReLU)
	bae = BinaryAutoEncoder(784, [128, 64, 128], [torch.max, torch.max, torch.max, torch.min])
	ooga = GeneticAlgTraining(ae)
	ooga_model = ooga.train()
	evaluation = Evaluation(ooga_model)
	
	#evaluation = Evaluation(ae, 1)
	#evaluation = Evaluation(fae, 1)
	#evaluation = Evaluation(mae, 80)
	#evaluation.train()
	evaluation.test()