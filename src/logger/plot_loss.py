import os
import matplotlib.pyplot as plt

from typing import List

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def plot_loss(losses: List[List[float]], save_path: str, name: str = 'default') -> None:
	for loss in losses:
		plt.plot(loss)
	plt.title(f'{name}')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.savefig(f'{save_path}/{name}.png')
	plt.clf()