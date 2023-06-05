from abc import abstractmethod
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from typing import Callable, List, Optional, Dict

from utils.metrics import Metrics
from globals.folder_names import LOG_FOLDER, IMAGE_FOLDER
from logger.output_to_csv import output_to_csv

class BaseTester:
	def __init__(
		self: "BaseTester",
		model: nn.Module,
		config: Dict,
		device: torch.device,
		data_loader: DataLoader,
		log_path: str,
		plotting: Optional[Callable[[List[torch.Tensor], List[torch.Tensor]], None]] = None,
		tensorboard_grpah: bool = False
	) -> None:
		self.model = model
		self.device = device
		self.data_loader = data_loader
		self.plotting = plotting
		self.tensorboard_graph = tensorboard_grpah
		self.config = config

		path_config = config['path']
		dataset = config['data']['dataset']
		self.csv_name = 'result.csv'
		self.experiment_name = f"{dataset}-{path_config['experiment_name']}"
	
		self.log_dir = os.path.join(log_path, LOG_FOLDER)
		self.image_dir = os.path.join(log_path, IMAGE_FOLDER)
		self.metrics: Metrics = Metrics(log_path)

	@abstractmethod
	def _test(self: "BaseTester", x: torch.Tensor) -> torch.Tensor:
		raise NotImplementedError

	def test(self: "BaseTester") -> Metrics:
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
			self.plotting(originals, results, save_path=self.image_dir, name=f"test-{self.experiment_name}")
		output_to_csv(originals, results, self.log_dir, f"reconstruction-test-{self.experiment_name}")

		if self.tensorboard_graph:
			writer = SummaryWriter(self.image_dir)
			a = next(iter(self.data_loader))
			writer.add_graph(self.model, a[0])
			writer.close()
		
		self.metrics.save(f"test-{self.csv_name}")
		return self.metrics

			

