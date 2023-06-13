import torch
from functools import partial
from enum import Enum

def null_operator(x: torch.Tensor) -> torch.Tensor:
	return torch.zeros(1)

def co_null_operator(x: torch.Tensor) -> torch.Tensor:
	return torch.ones(1)

def fuzzy_min(x: torch.Tensor) -> torch.Tensor:
	return torch.min(x, dim=-1).values

def fuzzy_max(x: torch.Tensor) -> torch.Tensor:
	return torch.max(x, dim=-1).values

def fuzzy_not(x: torch.Tensor) -> torch.Tensor:
	return 1 - x

# algebraic t-norm
def fuzzy_alg(x: torch.Tensor) -> torch.Tensor:
	return x.prod(dim=-1)

# algebraic t-conorm
def fuzzy_coalg(x: torch.Tensor) -> torch.Tensor:
	result = torch.zeros(x.shape[0], x.shape[1]).to(x.get_device())
	for i, sample in enumerate(x):
		for u, row in enumerate(sample):
			agg = row[0]
			for elem in row[1:]:
				agg = agg + elem - agg * elem
			result[i][u] = agg
	return result

# Lukasiewicz t-norm
def fuzzy_luk(x: torch.Tensor) -> torch.Tensor:
	result = torch.zeros(x.shape[0], x.shape[1])
	for i, sample in enumerate(x):
		for u, row in enumerate(sample):
			agg = row[0]
			for elem in row[1:]:
				agg = torch.max(torch.Tensor([(agg + elem - 1), 0]))
			result[i][u] = agg
	return result


# Lukasiewicz t-conorm
def fuzzy_coluk(x: torch.Tensor) -> torch.Tensor:
	result = torch.zeros(x.shape[0], x.shape[1])
	for i, sample in enumerate(x):
		for u, row in enumerate(sample):
			agg = row[0]
			for elem in row[1:]:
				agg = torch.max(torch.Tensor([(agg + elem), 1]))
			result[i][u] = agg
	return result

# Einstein t-norm
def fuzzy_ein(x: torch.Tensor) -> torch.Tensor:
	result = torch.zeros(x.shape[0], x.shape[1])
	for i, sample in enumerate(x):
		for u, row in enumerate(sample):
			agg = row[0]
			for elem in row[1:]:
				(elem * agg) / (2 - (elem + agg - elem * agg))
			result[i][u] = agg
	return result

# Einstein t-conorm
def fuzzy_coein(x: torch.Tensor) -> torch.Tensor:
	result = torch.zeros(x.shape[0], x.shape[1])
	for i, sample in enumerate(x):
		for u, row in enumerate(sample):
			agg = row[0]
			for elem in row[1:]:
				(elem + agg) / (1 + elem * agg)
			result[i][u] = agg
	return result

class T_Norm(Enum):
	min = partial(fuzzy_min)
	alg = partial(fuzzy_alg)
	luk = partial(fuzzy_luk)
	ein = partial(fuzzy_ein)
	Not = partial(fuzzy_not)
	null = partial(null_operator)

class T_Conorm(Enum):
	max = partial(fuzzy_max)
	alg = partial(fuzzy_coalg)
	luk = partial(fuzzy_coluk)
	ein = partial(fuzzy_coein)
	Not = partial(fuzzy_not)
	null = partial(co_null_operator)