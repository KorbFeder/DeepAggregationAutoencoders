import torch

def tMin(x: torch.Tensor):
	return torch.min(x)
	
def tProduct(x: torch.Tensor):
	# todo -> problem when more than one operand
	return torch.prod(x)

def tLukasiewicz(x: torch.Tensor):
	# todo for multiple operands
	#return torch.max()
	pass