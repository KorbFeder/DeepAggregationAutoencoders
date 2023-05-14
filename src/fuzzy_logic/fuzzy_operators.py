import torch

def null_operator(x: torch.Tensor):
	return torch.zeros(1)

def fuzzy_min(x: torch.Tensor, dim: int = 0):
	return torch.min(x, dim)

def fuzzy_max(x: torch.Tensor, dim: int = 0):
	return torch.max(x, dim)

def fuzzy_not(x: torch.Tensor):
	return 1 - x

#def fuzzy_min(x: torch.Tensor):
#	return torch.prod(x)