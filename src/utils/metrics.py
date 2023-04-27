import pandas as pd
import json
import os
from time import time

from typing import TypedDict, List

from logger.plot_loss import plot_loss

MetricType = TypedDict(
	'MetricType',
	{
		'loss': List[float],
		'total_num_of_samples': int,
		'time_elapsed': float,
		'epochs': int, 
		'iterations': int
	}
)

class Metrics:
	def __init__(self: "Metrics") -> None:
		self.reset()

	def add(self: "Metrics", epoch: int, processed_samples: int, loss: List[float]):
		curr_time = time()
		elapsed_time = curr_time - self.first_time
		self.total_num_of_samples += processed_samples
		self.curr_iteration += 1

		self.metrics.append({
			'loss': loss,
			'total_num_of_samples': self.total_num_of_samples,
			'time_elapsed': elapsed_time,
			'epochs': epoch, 
			'iterations': self.curr_iteration
		})

	def reset(self: "Metrics"):
		self.metrics: List[MetricType] = []
		self.first_time = time()
		self.total_num_of_samples = 0
		self.curr_iteration = 0

	def print_last(self: "Metrics") -> None:
		print(json.dumps(self.metrics[-1], indent=2, sort_keys=True))

	def plot_loss(self: "Metrics", name: str = 'metric-loss') -> None:
		losses = list(map(list, zip(*[metric['loss'] for metric in self.metrics])))
		for loss in losses:
			plot_loss(loss, name)

	def save(self: "Metrics", save_path: str, name: str = 'results.csv') -> None:
		df = pd.DataFrame(self.metrics)
		df.to_csv(os.path.join(save_path, name), index=False)
