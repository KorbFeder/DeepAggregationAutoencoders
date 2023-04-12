import torch
import numpy as np

from .Membership import Membership

class Fuzzyification(object):
	def __init__(self: "Fuzzyification", membership: Membership, center: float = 0):
		self.membership = membership
		self.center = center	
		
	def __call__(self: "Fuzzyification", sample: torch.Tensor) -> torch.Tensor:
		return np.vectorize(self.membership(sample.min(), sample.max(), self.center))(sample)
		
