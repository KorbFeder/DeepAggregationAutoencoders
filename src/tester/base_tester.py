from abc import abstractmethod
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from typing import Callable, List, Optional, Dict

from utils.metrics import Metrics
from logger.output_to_csv import output_to_csv

class BaseTester:
	def __init__(
		self: "BaseTester",
		model: nn.Module,
		config: Dict,
		device: torch.device,
		data_loader: DataLoader,
		plotting: Optional[Callable[[List[torch.Tensor], List[torch.Tensor]], None]] = None
	) -> None:
		self.model = model
		self.device = device
		self.data_loader = data_loader
		self.plotting = plotting
		self.config = config

		path_config = config['path']
		dataset = config['data']['dataset']
		self.csv_save_path = path_config['csv_save_path']
		self.csv_name = f"{dataset}-{path_config['csv_name']}"
		self.plot_save_path = path_config['plot_save_path']
		self.plot_name = f"{dataset}-{path_config['plot_name']}"
	
		self.metrics: Metrics = Metrics()

	@abstractmethod
	def _test(self: "BaseTester", x: torch.Tensor) -> torch.Tensor:
		raise NotImplementedError

	def test(self: "BaseTester"):
		print('started testing on test dataset ...')
		results = []
		originals = []
		error = []

		for x, _ in tqdm(self.data_loader):
			with torch.no_grad():
				x = x.to(self.device)

				result = self._test(x)
				error = nn.MSELoss()(x, result)
				results.append(result.cpu())
				originals.append(x.cpu())
				self.metrics.add(1, len(x), [error.cpu().item()])
		
		if self.plotting:
			self.plotting(originals, results, save_path=self.plot_save_path, name=f"test-{self.plot_name}")
		output_to_csv(originals, results, self.csv_save_path, f"test-{self.plot_name}")
		
		self.metrics.per_sample_loss(self.plot_save_path, f"test-{self.plot_name}")
		self.metrics.save(self.csv_save_path, f"test-{self.csv_name}")

			

