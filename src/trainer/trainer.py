import torch
from torch import nn
from torch.optim import Adam

from typing import Callable, Optional
from torch.utils.data import DataLoader

from model.AutoEncoder import AutoEncoder
from model.DeepAggregateAutoEncoder import DeepAggregateAutoEncoder
from model.MinMaxAutoEncoder import MinMaxAutoEncoder
from data_loader.Datafetcher import Datafetcher
from logger.plotting import progress_bar
from logger.plot_loss import plot_loss


class Trainer:
	def __init__(
		self: "Trainer", 
		model: nn.Module, 
		data_fetcher: Datafetcher, 
		epochs: int = 1, 
		plot_outputs: Callable[[torch.Tensor, torch.Tensor], None] = None,
		error = nn.MSELoss(),
		run_name = 'default',
		seed: Optional[int] = None,
		backtransformation: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
	) -> None:
		self.model = model
		self.epochs = epochs
		self.error = error
		self.plot_outputs = plot_outputs
		self.run_name = run_name
		self.train_loader: DataLoader = data_fetcher.get_train_dataloader()
		self.test_loader: DataLoader = data_fetcher.get_test_dataloader()
		self.backtranformation = backtransformation

		if seed != None:
			self.set_seeds(seed)

	def train(self: "Trainer") -> None:
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
				plot_loss(losses, self.run_name)


	def test(self: "Trainer") -> None:
		flatten = nn.Flatten()
		originals = []
		outputs = []
		loss = []
		for i, batch_features in enumerate(self.test_loader):

			batch_features = flatten(batch_features)

			with torch.no_grad():
				output = self.model(batch_features)

			loss.append(self.error(output, batch_features).item())

			originals += self.backtranformation(batch_features)
			outputs += self.backtranformation(output)
			progress_bar(i, len(self.test_loader))

		avg_loss = sum(loss) / len(loss)
		print(f"test-avg-loss = {avg_loss}")
		if self.plot_outputs:
			self.plot_outputs(originals, outputs)

	def set_seeds(self: "Trainer", seed: int):
		torch.manual_seed(seed)

	