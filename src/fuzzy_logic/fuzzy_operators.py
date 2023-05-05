import torch

def null_operator(x: torch.Tensor):
	return torch.zeros(1)

def fuzzy_min(x: torch.Tensor):
	return torch.min(x)

def fuzzy_max(x: torch.Tensor):
	return torch.max(x)

def fuzzy_not(x: torch.Tensor):
	return 1 - x

#def fuzzy_min(x: torch.Tensor):
#	return torch.prod(x)