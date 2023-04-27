import torch
from typing import List

def progress_bar(current, total, bar_length=20) -> None:
    fraction = current / total

    arrow = int(fraction * bar_length - 1) * '-' + '>'
    padding = int(bar_length - len(arrow)) * ' '

    ending = '\n' if current == total else '\r'

    print(f'Progress: [{arrow}{padding}] {int(fraction*100)}%', end=ending)

def plot_wine_results(originals: List[torch.Tensor], outputs: List[torch.Tensor], rows: int = 8):
	for i in range(rows):
		print(f"original: {originals[i]}\noutputs: {outputs[i]}")