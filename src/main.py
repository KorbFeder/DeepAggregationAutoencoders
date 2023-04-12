import torch
import torch.nn as nn
from models.AutoEncoder import AutoEncoder
from models.DeepAggregateAutoEncoder import DeepAggregateAutoEncoder
from Evaluation import Evaluation

if __name__ == "__main__":
	fae = DeepAggregateAutoEncoder(784, [128, 64, 128], [torch.min, torch.max, torch.max, torch.min], activation=nn.ReLU)
	ae = AutoEncoder(784, [128, 64, 128])
	#evaluation = Evaluation(ae, 1)
	evaluation = Evaluation(fae, 1)
	evaluation.train()
	evaluation.test()