import os
import matplotlib.pyplot as plt
import numpy as np

from typing import List, Optional

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def plot_loss(losses: List[List[float]], save_path: str, name: str = 'default', y_label = 'Epochs', legend: Optional[List[str]] = None) -> None:
	for i, loss in enumerate(losses):
		label = None
		if legend != None:
			label = legend[i]

		plt.plot(loss, label=label)

	#ax = plt.gca()
	#ax.set_ylim([0, np.min([np.max(losses), 1])])
	plt.title(f'{name}')
	plt.xlabel(y_label)
	plt.ylabel('Loss')

	if legend:
		plt.legend()
	
	plt.savefig(os.path.join(save_path, f'{name}.png'))
	plt.clf()