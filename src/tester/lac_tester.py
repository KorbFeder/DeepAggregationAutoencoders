import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from typing import Callable, List, Optional

from tester.base_tester import BaseTester



class LacTester(BaseTester):
	def __init__(
		self: "BaseTester", 
		model: nn.Module, 
		device: torch.device, 
		data_loader: DataLoader, 
		plotting: Optional[Callable[[List[torch.Tensor]], None]] = None
	) -> None:
		super().__init__(model, device, data_loader, plotting)

	def _test(self: "BaseTester", x: torch.Tensor) -> torch.Tensor:
		_, x = self.model(x)
		return x