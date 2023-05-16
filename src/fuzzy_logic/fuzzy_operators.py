import torch

def null_operator(x: torch.Tensor) -> torch.Tensor:
	return torch.zeros(1)

def fuzzy_min(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
	return torch.min(x, dim).values

def fuzzy_max(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
	return torch.max(x, dim).values

def fuzzy_not(x: torch.Tensor) -> torch.Tensor:
	return 1 - x

# algebraic t-norm
def fuzzy_alg(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
	return x.prod(dim)

# algebraic t-conorm
def fuzzy_coalg(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
	result = torch.zeros(x.shape[0], x.shape[1])
	for i, sample in enumerate(x):
		for u, row in enumerate(sample):
			agg = row[0]
			for elem in row[1:]:
				agg = agg + elem - agg * elem
			result[i][u] = agg
	return agg

# Lukasiewicz t-norm
def fuzzy_luk(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
	result = torch.zeros(x.shape[0], x.shape[1])
	for i, sample in enumerate(x):
		for u, row in enumerate(sample):
			agg = row[0]
			for elem in row[1:]:
				agg = torch.max(torch.Tensor([(agg + elem - 1), 0]))
			result[i][u] = agg
	return agg


# Lukasiewicz t-conorm
def fuzzy_coluk(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
	result = torch.zeros(x.shape[0], x.shape[1])
	for i, sample in enumerate(x):
		for u, row in enumerate(sample):
			agg = row[0]
			for elem in row[1:]:
				agg = torch.max(torch.Tensor([(agg + elem), 1]))
			result[i][u] = agg
	return agg

# Einstein t-norm
def fuzzy_ein(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
	result = torch.zeros(x.shape[0], x.shape[1])
	for i, sample in enumerate(x):
		for u, row in enumerate(sample):
			agg = row[0]
			for elem in row[1:]:
				(elem * agg) / (2 - (elem + agg - elem * agg))
			result[i][u] = agg
	return agg

# Einstein t-conorm
def fuzzy_coein(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
	result = torch.zeros(x.shape[0], x.shape[1])
	for i, sample in enumerate(x):
		for u, row in enumerate(sample):
			agg = row[0]
			for elem in row[1:]:
				(elem + agg) / (1 + elem * agg)
			result[i][u] = agg
	return agg


#def fuzzy_min(x: torch.Tensor):
#	return torch.prod(x)