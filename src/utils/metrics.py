import pandas as pd
import json
from time import time

from typing import TypedDict, List

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
		self.metrics: List[MetricType] = []
		self.last_time = time()
		self.total_num_of_samples = 0
		self.curr_iteration = 0

	def add(self: "Metrics", epoch: int, processed_samples: int, loss: List[float]):
		curr_time = time()
		elapsed_time = curr_time - self.last_time
		self.last_time = curr_time
		self.total_num_of_samples += processed_samples
		self.curr_iteration += 1

		self.metrics.append({
			'loss': loss,
			'total_num_of_samples': self.total_num_of_samples,
			'time_elapsed': elapsed_time,
			'epochs': epoch, 
			'iterations': self.curr_iteration
		})

	def print(self: "Metrics") -> None:
		print(json.dumps(self.metrics[-1], indent=2, sort_keys=True))

	def save(self: "Metrics", save_path: str) -> None:
		df = pd.DataFrame(self.metrics)
		df.to_csv(save_path, index=False)
