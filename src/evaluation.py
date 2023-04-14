import torch
from torch import nn
from torch.optim import Adam

from typing import Callable, Tuple
from torch.utils.data import DataLoader

from models.AutoEncoder import AutoEncoder
from models.DeepAggregateAutoEncoder import DeepAggregateAutoEncoder
from models.MinMaxAutoEncoder import MinMaxAutoEncoder
from utils.plotting import plot_outputs, progress_bar
from utils.data_loading import get_mnist_dataloaders
from fuzzy_logic.Fuzzyfication import Fuzzyification
from fuzzy_logic.Membership import Membership


class Evaluation:
	def __init__(
		self: "Evaluation", 
		model: nn.Module, 
		epochs: int, 
		load_data: Callable[[], Tuple[DataLoader, DataLoader]] = get_mnist_dataloaders, 
		error = nn.MSELoss()
	) -> None:
		self.model = model
		self.epochs = epochs
		self.error = error
		self.train_loader, self.test_loader = load_data()

	def train(self: "Evaluation") -> None:
		optim = Adam(self.model.parameters(), lr=1e-3)
		losses = []

		for epoch in range(self.epochs):
			loss = []
			for i, (batch_features, _) in enumerate(self.train_loader):

				batch_features = batch_features.view(-1, 784)

				optim.zero_grad()

				outputs = self.model(batch_features)

				train_loss = self.error(outputs, batch_features)

				train_loss.backward()
				optim.step()
				with torch.no_grad():
					loss.append(train_loss.item())
				
				progress_bar(i, len(self.train_loader))
			losses += loss

			loss = sum(losses) / len(self.train_loader)
			print(f"epoch: {epoch+1}/{self.epochs}, train-loss = {loss}")


	def test(self: "Evaluation") -> None:
		originals = []
		outputs = []
		loss = []
		for i, (batch_features, _) in enumerate(self.test_loader):

			batch_features = batch_features.view(-1, 784)

			with torch.no_grad():
				output = self.model(batch_features)

			loss.append(self.error(output, batch_features).item())

			originals += batch_features.reshape(-1, 28, 28)
			outputs += output.reshape(-1, 28, 28)
			progress_bar(i, len(self.test_loader))

		avg_loss = sum(loss) / len(self.test_loader)
		print(f"test-avg-loss = {avg_loss}")
		plot_outputs(originals, outputs)

if __name__ == "__main__":
	fae = DeepAggregateAutoEncoder(784, [128, 64, 128], [torch.min, torch.max, torch.max, torch.min], activation=nn.ReLU)
	ae = AutoEncoder(784, [128, 64, 128])
	mae = MinMaxAutoEncoder(784, [128, 64, 128], [torch.min, torch.max, torch.max, torch.min], activation=nn.ReLU)
	#evaluation = Evaluation(ae, 1)
	#evaluation = Evaluation(fae, 1)
	evaluation = Evaluation(mae, 1)
	evaluation.train()
	evaluation.test()




	