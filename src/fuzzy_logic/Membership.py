import torch
from abc import ABC, abstractclassmethod

class Membership(ABC):
	@abstractclassmethod
	def fuzzification(self: "Membership", input_data: torch.Tensor) -> torch.Tensor:
		pass

	@abstractclassmethod
	def defuzzification(self: "Membership", input_data: torch.Tensor) -> torch.Tensor:
		pass