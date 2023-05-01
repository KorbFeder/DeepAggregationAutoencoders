import torch
import matplotlib.pyplot as plt
from typing import List

def plot_mnist_outputs(originals: List[torch.Tensor], outputs: List[torch.Tensor], rows: int = 8, name: str = "mnist-output", show_plot: bool = False):		
	f, axarr = plt.subplots(4, rows, figsize=(12, 6))
	f.tight_layout()
	axarr[0, 0].set_title("original", fontsize=15)
	axarr[2, 0].set_title("generated", fontsize=15)
	for i in range(rows):
		axarr[0, i].imshow(originals[i][0].reshape(28, 28), cmap="gray")
		axarr[1, i].imshow(originals[i + rows][0].reshape(28, 28), cmap="gray")
		axarr[2, i].imshow(outputs[i][0].reshape(28, 28), cmap="gray")
		axarr[3, i].imshow(outputs[i + rows][0].reshape(28, 28), cmap="gray")
	if show_plot:
		plt.show()
	f.savefig(f"./image/{name}")
	plt.clf()