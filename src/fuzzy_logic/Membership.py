from enum import Enum
from typing import Callable

def medium_factory(domain_min: float, domain_max: float, center: float) -> Callable[[float], float]:
	def medium(x) -> float:
		if x <= domain_min:
			return 0
		elif domain_min < x and x <= center:
			return (x - domain_min) / (center - domain_min)
		elif center < x and x < domain_max:
			return 1 - ((x - center) / (domain_max - center))
		elif x >= domain_max:
			return 0
	return medium

def high_factory(domain_min: float, domain_max: float, _) -> Callable[[float], float]:
	def high(x: float) -> float: 
		if x < domain_min:
			return 0
		elif x > domain_max:
			return 1
		else: 
			return (x - domain_min) / (domain_max - domain_min)
	return high
	
def low_factory(domain_min: float, domain_max: float, _) -> Callable[[float], float]:
	def low(x: float) -> float: 
		if x < domain_min:
			return 1
		elif x > domain_max:
			return 0
		else: 
			return 1 - ((x - domain_min) / (domain_max - domain_min))
	return low

class Membership(Enum):
	low = low_factory
	medium = medium_factory
	high = high_factory