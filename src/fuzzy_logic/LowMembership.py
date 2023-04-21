import torch
from fuzzy_logic.Membership import Membership

class LowMembership(Membership):
	def __init__(self: "LowMembership") -> None:
		pass

	def fuzzification(self: "LowMembership", data: torch.Tensor) -> torch.Tensor:
		self.col_min = data.min(dim=0).values
		self.col_max = data.max(dim=0).values
		return self._low_membership(data)	

	def defuzzification(self: "LowMembership", result: torch.Tensor) -> torch.Tensor:
		return self._low_membership(result)


	def _low_membership(self: "LowMembership", input_data: torch.Tensor) -> torch.Tensor:
		output = torch.Tensor(input_data.shape)
		for i in range(len(input_data)):
			for u in range(len(input_data[i])):
				x = input_data[i][u]
				curr_min = self.col_min[u]
				curr_max = self.col_max[u]
				if x < curr_min:
					output[i][u] = 1
				elif x > curr_max:
					output[i][u] = 0
				else: 
					output[i][u] = 1 -(x - curr_min) / (curr_max - curr_min)
		return output

