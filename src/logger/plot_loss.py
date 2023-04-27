import matplotlib.pyplot as plt
from typing import List

def plot_loss(losses: List[float], name: str = 'default') -> None:
	plt.plot(losses)
	plt.title = 'Training Loss'
	plt.xlabel = 'Epochs'
	plt.ylabel = 'Loss'
	plt.savefig(f'./image/training_loss_curve-{name}.png')