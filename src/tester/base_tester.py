from abc import abstractmethod
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from typing import Callable, List, Optional

from utils.metrics import Metrics

class BaseTester:
	def __init__(
		self: "BaseTester",
		model: nn.Module,
		device: torch.device,
		data_loader: DataLoader,
		plotting: Optional[Callable[[List[torch.Tensor], List[torch.Tensor]], None]] = None
	) -> None:
		self.model = model
		self.device = device
		self.data_loader = data_loader
		self.plotting = plotting
		self.metrics = Metrics()

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
			self.plotting(originals, results)
		self.metrics.per_sample_loss('test')

			

