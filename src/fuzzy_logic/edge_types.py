import torch
from enum import Enum
from functools import partial

class EdgeType(Enum):
	no_edge = partial(lambda x: torch.zeros(x.shape))
	normal_edge = partial(lambda x: x)
	very = partial(lambda x: torch.square(x))
	somewhat = partial(lambda x: torch.sqrt(x + 1.e-8))
	Not = partial(lambda x: 1 - x)