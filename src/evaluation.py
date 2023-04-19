import torch
from torch import nn
from torch.optim import Adam

from typing import Callable, Tuple
from torch.utils.data import DataLoader

from models.AutoEncoder import AutoEncoder
from models.DeepAggregateAutoEncoder import DeepAggregateAutoEncoder
from models.MinMaxAutoEncoder import MinMaxAutoEncoder
from utils.plotting import plot_mnist_outputs, progress_bar
from utils.data_loading import get_mnist_dataloaders
from fuzzy_logic.Fuzzyfication import Fuzzyification
from fuzzy_logic.Membership import Membership


class Evaluation:
	def __init__(
		self: "Evaluation", 
		model: nn.Module, 
		epochs: int = 1, 
		load_data: Callable[[], Tuple[DataLoader, DataLoader]] = get_mnist_dataloaders, 
		plot_outputs: Callable[[torch.Tensor, torch.Tensor], None] = plot_mnist_outputs,
		error = nn.MSELoss()
	) -> None:
		self.model = model
		self.epochs = epochs
		self.error = error
		self.plot_outputs = plot_outputs
		self.train_loader, self.test_loader = load_data()

	def train(self: "Evaluation") -> None:
		optim = Adam(self.model.parameters(), lr=1e-3)
		flatten = nn.Flatten()
		losses = []
		originals = []
		outputs = []
	
		for epoch in range(self.epochs):
			loss = []
			for i, (batch_features, _) in enumerate(self.train_loader):

				batch_features = flatten(batch_features)

				optim.zero_grad()

				output = self.model(batch_features)

				train_loss = self.error(output, batch_features)

				train_loss.backward()
				optim.step()

				# save data for plotting
				with torch.no_grad():
					loss.append(train_loss.item())
					originals += batch_features
					outputs += output
					progress_bar(i, len(self.train_loader))

			# plot results
			with torch.no_grad():
				losses += loss
				loss = sum(loss) / len(self.train_loader)
				print(f"epoch: {epoch+1}/{self.epochs}, train-loss = {loss}")
				self.plot_outputs(originals, outputs, name=f"progress-{epoch+1}I{self.epochs}")


	def test(self: "Evaluation") -> None:
		originals = []
		outputs = []
		loss = []
		for i, (batch_features, _) in enumerate(self.test_loader):

			batch_features = batch_features.view(-1, 784)

			with torch.no_grad():
				output = self.model(batch_features)

			loss.append(self.error(output, batch_features).item())

			originals += batch_features
			outputs += output
			progress_bar(i, len(self.test_loader))

		avg_loss = sum(loss) / len(self.test_loader)
		print(f"test-avg-loss = {avg_loss}")
		self.plot_outputs(originals, outputs)

if __name__ == "__main__":
	fae = DeepAggregateAutoEncoder(784, [128, 64, 128], [torch.min, torch.max, torch.max, torch.min], activation=nn.ReLU)
	ae = AutoEncoder(784, [128, 64, 128])
	mae = MinMaxAutoEncoder(784, [128, 64, 128], [torch.min, torch.max, torch.max, torch.min], activation=nn.ReLU)
	#evaluation = Evaluation(ae, 1)
	#evaluation = Evaluation(fae, 1)
	evaluation = Evaluation(mae, 1)
	evaluation.train()
	evaluation.test()




	