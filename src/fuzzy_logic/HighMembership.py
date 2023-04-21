import torch
from fuzzy_logic.Membership import Membership

class HighMembership(Membership):
	def __init__(self: "HighMembership", data: torch.Tensor) -> None:
		self.data = data
		self.col_min = data.min(dim=0).values
		self.col_max = data.max(dim=0).values

	def fuzzification(self: "HighMembership"):
		return self._high_membership(self.data)	

	def defuzzification(self: "HighMembership", result: torch.Tensor):
		return self._high_membership(result)


	def _high_membership(self: "HighMembership", input_data: torch.Tensor):
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