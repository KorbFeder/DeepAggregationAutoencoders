import torch
import numpy as np

from .Membership import Membership

class Defuzzyfication(object):
	def __init__(self: "Defuzzyfication", membership: Membership,  domain_min: float, domain_max: float, center: float = 0):
		self.membership = membership
		self.domain_min = domain_min
		self.domain_max = domain_max
		self.center = center

	def __call__(self: "Defuzzyfication", sample: torch.Tensor) -> torch.Tensor:
		return np.vectorize(self.membership(self.domain_min, self.domain_max, self.center))(sample)

