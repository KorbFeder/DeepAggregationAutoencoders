import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from typing import List

from data_loader.wine_data_loader import scaler

def plot_wine_results(
	originals: List[torch.Tensor], 
	outputs: List[torch.Tensor], 
	name: str, 
	save_path: str = "./images",
	rows = 20
) -> None:
	org = torch.cat(originals).numpy()
	originals = np.round(scaler.inverse_transform(org), 6)
	out = torch.cat(outputs).numpy()
	outputs = np.round(scaler.inverse_transform(out), 6)

	column_names = [
		"facid", "vacid", "cacid", 
		"sugar", "chlor", "fsulfur", 
		"tsulfur", "dens", "pH", 
		"sulph", "alc", "qual"
	]

	original_cols = [f"org-{name}" for name in column_names]
	output_cols = [f"gen-{name}" for name in column_names]

	fig, ax = plt.subplots(figsize=(20, 10))
	df_originals = pd.DataFrame(originals, columns=original_cols)
	df_outputs = pd.DataFrame(outputs, columns=output_cols)
	df = pd.concat([df_originals, df_outputs], axis=1)
	table_df = df[:rows]
	table = ax.table(cellText=table_df.values, colLabels=table_df.columns, loc='center')
	#table.set_fontsize(16)
	plt.savefig(f'{save_path}/{name}-table')
	plt.clf()