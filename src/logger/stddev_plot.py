import os
import torch
import matplotlib.pyplot as plt

from typing import List, Optional

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def stddev_plot(losses: List[List[float]], save_path: str, name: str = 'default', y_label = 'Epochs') -> None:
	losses = torch.Tensor(losses)
	mean_loss = torch.mean(losses, dim=0)
	std_deviation = torch.std(losses, dim=0)

	x = range(len(mean_loss))
	plt.plot(x, mean_loss)

	plt.fill_between(x, mean_loss - std_deviation, mean_loss + std_deviation, alpha=0.2)

	plt.title(f'{name}')
	plt.xlabel(y_label)
	plt.ylabel('Loss')

	plt.savefig(os.path.join(save_path, f'{name}.png'))
	plt.clf()