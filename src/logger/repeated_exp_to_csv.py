from typing import List
from utils.metrics import Metrics
import pandas as pd
import os

def repeated_exp_to_csv(train_metrics: List[List[Metrics]], test_metrics: List[List[Metrics]], save_path: str):
	data = {}
	for u, _metrics in enumerate(train_metrics):
		for i, run in enumerate(_metrics):
			data[f'train-exp{u}-run{i}'] = [m['loss'] for m in run.metrics]
	for u, _metrics in enumerate(test_metrics):
		for i, run in enumerate(_metrics):
			data[f'test-exp{u}-run{i}'] = [m['loss'] for m in run.metrics]
	
	df = pd.DataFrame(data)
	df.to_csv(os.path.join(save_path, 'all_losses.csv'))
	return df


