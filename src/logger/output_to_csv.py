import torch
import matplotlib.pyplot as plt
import pandas as pd

from typing import List

def output_to_csv(originals: List[torch.Tensor], outputs: List[torch.Tensor],save_path: str, name: str) -> None:
	length = originals[0].shape[1]
	originals = torch.cat(originals).numpy()
	outputs = torch.cat(outputs).numpy()

	orig_columns = [f"original-{i}" for i in range(length)]
	gen_columns = [f"generated-{i}" for i in range(length)]

	fig, ax = plt.subplots(figsize=(20, 10))
	df_originals = pd.DataFrame(originals, columns=orig_columns)
	df_outputs = pd.DataFrame(outputs, columns=gen_columns)
	df = pd.concat([df_originals, df_outputs], axis=1)

	df.to_csv(f'{save_path}/{name}.csv')
