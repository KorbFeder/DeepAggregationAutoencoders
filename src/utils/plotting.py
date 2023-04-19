import torch
import matplotlib.pyplot as plt
from typing import List

def plot_outputs(originals: List[torch.Tensor], outputs: List[torch.Tensor], rows: int = 8, name: str = "mnist-output", show_plot: bool = False):		
	f, axarr = plt.subplots(4, rows, figsize=(12, 6))
	f.tight_layout()
	axarr[0, 0].set_title("original", fontsize=15)
	axarr[2, 0].set_title("generated", fontsize=15)
	for i in range(rows):
		axarr[0, i].imshow(originals[i])
		axarr[1, i].imshow(originals[i + rows])
		axarr[2, i].imshow(outputs[i])
		axarr[3, i].imshow(outputs[i + rows])
	if show_plot:
		plt.show()
	f.savefig(f"./images/{name}")


def progress_bar(current, total, bar_length=20):
    fraction = current / total

    arrow = int(fraction * bar_length - 1) * '-' + '>'
    padding = int(bar_length - len(arrow)) * ' '

    ending = '\n' if current == total else '\r'

    print(f'Progress: [{arrow}{padding}] {int(fraction*100)}%', end=ending)