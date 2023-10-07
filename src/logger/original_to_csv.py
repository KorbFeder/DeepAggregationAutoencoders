import torch
import pandas as pd
import numpy as np

from typing import List

from data_loader.csv_data_loader import scaler
from sklearn.utils.validation import check_is_fitted

def original_to_csv(originals: List[torch.Tensor], outputs: List[torch.Tensor],save_path: str, name: str) -> None:
	try:
		length = originals[0].shape[1]
		originals = torch.cat(originals).numpy()
		originals = np.round(scaler.inverse_transform(originals), 6)
		outputs = torch.cat(outputs).numpy()
		outputs = np.round(scaler.inverse_transform(outputs), 6)

		orig_columns = [f"original-{i}" for i in range(length)]
		gen_columns = [f"generated-{i}" for i in range(length)]

		df_originals = pd.DataFrame(originals, columns=orig_columns)
		df_outputs = pd.DataFrame(outputs, columns=gen_columns)
		df = pd.concat([df_originals, df_outputs], axis=1)

		df.to_csv(f'{save_path}/{name}.csv')
	except:
		return

