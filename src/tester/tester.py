import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from typing import Callable, List, Optional, Dict

from tester.base_tester import BaseTester

class Tester(BaseTester):
	def __init__(
		self: "BaseTester", 
		model: nn.Module, 
		config: Dict,
		device: torch.device, 
		data_loader: DataLoader, 
		plotting: Optional[Callable[[List[torch.Tensor]], None]] = None
	) -> None:
		super().__init__(model, config, device, data_loader, plotting)

	def _test(self: "BaseTester", x: torch.Tensor) -> torch.Tensor:
		x = self.model(x)
		return x