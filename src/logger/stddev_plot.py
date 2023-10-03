import os
import torch
import matplotlib.pyplot as plt
import matplotlib.pyplot

from typing import List, Optional

os.environ['KMP_DUPLICATE_LIB_OK']='True'
matplotlib.pyplot.locator_params(nbins=10)
NR_LABELS = 10
#EDGE_COLOR = "#232323"

LMU_DARK_GREY = "#626468"
LMU_GREEN = '#00883A'
LMU_LIGHT_GREY = "#E6E6E7"
LMU_BLAU = "#009FE3"
LMU_VIOLETT = "#8C4091"

TOP_CUTOFF = 0.2

colors = [LMU_DARK_GREY, LMU_GREEN, LMU_BLAU, LMU_VIOLETT]

def stddev_plot(_losses: List[List[List[float]]], save_path: str, name: str = 'default', x_label = 'Epochs', x_ticks = None, 
		use_bands=True, legend = None) -> None:
	fig = plt.figure()
	for i, losses in enumerate(_losses):
		losses = torch.Tensor(losses)
		mean_loss = torch.mean(losses, dim=0)
		std_deviation = torch.std(losses, dim=0)

		color = None
		if i < len(colors):
			color = colors[i]
		x = range(len(mean_loss))
		if legend:
			if color:
				plt.plot(x, mean_loss, label=legend[i], color=color)
			else:
				plt.plot(x, mean_loss, label=legend[i])
		else:
			if color:
				plt.plot(x, mean_loss, color=color)
			else:
				plt.plot(x, mean_loss)


		lower = mean_loss - std_deviation
		lower = list(map(lambda m: 0 if m < 0 else m, lower))
		if use_bands:
			if color:
				plt.fill_between(x, lower, mean_loss + std_deviation, alpha=0.2, color=color)
			else:
				plt.fill_between(x, lower, mean_loss + std_deviation, alpha=0.2)

	plt.xlabel(x_label)
	plt.ylabel('Loss')
	if legend:
		plt.legend(loc="upper right")
	step_size = int(len(mean_loss) / NR_LABELS)
	if x_ticks:
		a = []
		a_ticks = []
		for i in range(NR_LABELS):
			a.append(x[i*step_size])
			a_ticks.append(x_ticks[i*step_size])
		plt.xticks(a, a_ticks)

	bands = ''
	if use_bands:
		bands = '-std'

	fig.tight_layout()
	plt.yscale('log')
	fig.savefig(os.path.join(save_path, f'{name}{bands}-log.png'))
	plt.yscale('linear')
	fig.savefig(os.path.join(save_path, f'{name}{bands}.png'))
	plt.ylim(0, TOP_CUTOFF)
	fig.savefig(os.path.join(save_path, f'{name}{bands}-cropped.png'))
	plt.clf()