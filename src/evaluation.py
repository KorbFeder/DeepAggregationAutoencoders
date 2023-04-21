import torch
from torch import nn
from torch.optim import Adam

from typing import Callable, Optional
from torch.utils.data import DataLoader

from models.AutoEncoder import AutoEncoder
from models.DeepAggregateAutoEncoder import DeepAggregateAutoEncoder
from models.MinMaxAutoEncoder import MinMaxAutoEncoder
from data.Datafetcher import Datafetcher
from utils.plotting import progress_bar, plot_training_loss
from fuzzy_logic.Fuzzyfication import Fuzzyification
from fuzzy_logic.Membership import Membership


class Evaluation:
	def __init__(
		self: "Evaluation", 
		model: nn.Module, 
		data_fetcher: Datafetcher, 
		epochs: int = 1, 
		plot_outputs: Callable[[torch.Tensor, torch.Tensor], None] = None,
		error = nn.MSELoss(),
		run_name = 'default',
		seed: Optional[int] = None
	) -> None:
		self.model = model
		self.epochs = epochs
		self.error = error
		self.plot_outputs = plot_outputs
		self.run_name = run_name
		self.train_loader: DataLoader = data_fetcher.get_train_dataloader()
		self.test_loader: DataLoader = data_fetcher.get_test_dataloader()

		if seed != None:
			self.set_seeds(seed)

	def train(self: "Evaluation") -> None:
		optim = Adam(self.model.parameters(), lr=1e-3)
		flatten = nn.Flatten()
		losses = []
		originals = []
		outputs = []
	
		for epoch in range(self.epochs):
			loss = []
			for i, batch_features in enumerate(self.train_loader):

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

			# results after each epoch
			with torch.no_grad():
				loss = sum(loss) / len(loss)
				losses.append(loss)
				print(f"epoch: {epoch+1}/{self.epochs}, train-loss = {loss}")
			
		# plot results after end of training
		with torch.no_grad():
			if self.epochs > 1:
				plot_training_loss(losses, self.run_name)


	def test(self: "Evaluation") -> None:
		flatten = nn.Flatten()
		originals = []
		outputs = []
		loss = []
		for i, batch_features in enumerate(self.test_loader):

			batch_features = flatten(batch_features)

			with torch.no_grad():
				output = self.model(batch_features)

			loss.append(self.error(output, batch_features).item())

			originals += batch_features
			outputs += output
			progress_bar(i, len(self.test_loader))

		avg_loss = sum(loss) / len(loss)
		print(f"test-avg-loss = {avg_loss}")
		if self.plot_outputs:
			self.plot_outputs(originals, outputs)

	def set_seeds(self: "Evaluation", seed: int):
		torch.manual_seed(seed)

	