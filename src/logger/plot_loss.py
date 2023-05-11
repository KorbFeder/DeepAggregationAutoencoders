import os
import matplotlib.pyplot as plt

from typing import List

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def plot_loss(losses: List[List[float]], save_path: str, name: str = 'default', y_label = 'Epochs') -> None:
	for loss in losses:
		plt.plot(loss)
	plt.title(f'{name}')
	plt.xlabel(y_label)
	plt.ylabel('Loss')
	plt.savefig(f'{save_path}/{name}.png')
	plt.clf()