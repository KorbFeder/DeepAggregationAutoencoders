import pandas as pd
import json
import os
from itertools import groupby
from statistics import mean
from time import time

from typing import TypedDict, List

from logger.plot_loss import plot_loss
from globals.folder_names import IMAGE_FOLDER, LOG_FOLDER

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
	def __init__(self: "Metrics", save_path: str) -> None:
		self.save_path = save_path
		self.image_save_path = os.path.join(save_path, IMAGE_FOLDER)
		self.log_save_path = os.path.join(save_path, LOG_FOLDER)
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

	def reset(self: "Metrics") -> None:
		self.metrics: List[MetricType] = []
		self.first_time = time()
		self.total_num_of_samples = 0
		self.curr_iteration = 0

	@property
	def episodic_loss(self: "Metrics") -> List[List[float]]:
		groups = groupby(self.metrics, key=lambda x: x['epochs'])
		avg_losses = []

		for _, group in groups:
			group = list(group)
			ep_losses = list(map(list, zip(*[metric['loss'] for metric in group])))
			avg_losses.append([mean(loss) for loss in ep_losses])

		return list(map(list, zip(*avg_losses)))
	
	@property
	def per_sample_loss(self: "Metrics") -> List[List[float]]:
		return list(map(list, zip(*[metric['loss'] for metric in self.metrics])))

	@property
	def curr_time_elapsed(self: "Metrics") -> float:
		return self.metrics[-1]['time_elapsed']

	def print_last(self: "Metrics") -> None:
		print(json.dumps(self.metrics[-1], indent=2, sort_keys=True))
	
	def print_avg_loss(self: "Metrics"):
		print(f'\t{[mean(loss) for loss in self.per_sample_loss]}')

	def plot_loss(self: "Metrics", name: str = 'train') -> None:
		self.plot_episodic_loss(name)
		self.plot_per_sample_loss(name)

	def plot_per_sample_loss(self: "Metrics", name: str) -> None:
		name = name + "-per-sample-loss"
		losses = self.per_sample_loss
		plot_loss(losses, self.image_save_path, name, y_label='Samples')
	
	def plot_episodic_loss(self: "Metrics", name: str) -> None:
		name = name + "-episodic-loss"
		losses = self.episodic_loss

		if len(losses[0]) <= 1:
			return

		plot_loss(losses, self.image_save_path, name)
	
	def save(self: "Metrics", name: str = 'results.csv') -> None:
		df = pd.DataFrame(self.metrics)
		df.to_csv(os.path.join(self.log_save_path, name), index=False)
