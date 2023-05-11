import torch
from typing import List

def plot_wine_results(originals: List[torch.Tensor], outputs: List[torch.Tensor], name: str, save_path: str,  rows: int = 8):
	for i in range(rows):
		print(f"original: {originals[i]}\noutputs: {outputs[i]}")