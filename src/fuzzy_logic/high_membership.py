import torch
from fuzzy_logic.membership import Membership

class HighMembership(Membership):
	def fuzzification(self: "HighMembership", data: torch.Tensor) -> torch.Tensor:
		self.col_min = data.min(dim=0).values
		self.col_max = data.max(dim=0).values
		return self._high_membership(data)	

	def defuzzification(self: "HighMembership", input_data: torch.Tensor) -> torch.Tensor:
		output = torch.Tensor(input_data.shape)
		for i in range(len(input_data)):
			for u in range(len(input_data[i])):
				x = input_data[i][u]
				curr_min = self.col_min[u]
				curr_max = self.col_max[u]
				if x < 0:
					output[i][u] = curr_min
				elif x > 1:
					output[i][u] = curr_max
				else: 
					output[i][u] = x * (curr_max - curr_min) + curr_min 
		return output

	def _high_membership(self: "HighMembership", input_data: torch.Tensor) -> torch.Tensor:
		output = torch.Tensor(input_data.shape)
		for i in range(len(input_data)):
			for u in range(len(input_data[i])):
				x = input_data[i][u]
				curr_min = self.col_min[u]
				curr_max = self.col_max[u]
				if x < curr_min:
					output[i][u] = 0
				elif x > curr_max:
					output[i][u] = 1
				else: 
					output[i][u] = (x - curr_min) / (curr_max - curr_min)
		return output