import matplotlib.pyplot as plt
from typing import List

def plot_loss(losses: List[List[float]], name: str = 'default') -> None:
	for loss in losses:
		plt.plot(loss)
	plt.title(f'{name}')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.savefig(f'./image/{name}.png')
	plt.clf()