import torch
import numpy as np
import random

from typing import Optional

def set_random_seed(seed: Optional[int]):
	if seed:
		torch.manual_seed(seed)
		np.random.seed(seed)
		random.seed(seed)

