import torch
from fuzzy_logic.Membership import Membership

class MediumMembership(Membership):
	def __init__(self: "MediumMembership", data: torch.Tensor) -> None:
		self.data = data
		self.col_min = data.min(dim=0).values
		self.col_max = data.max(dim=0).values

	def fuzzification(self: "MediumMembership"):
		return self._medium_membership(self.data)	

	def defuzzification(self: "MediumMembership", result: torch.Tensor):
		return self._medium_membership(result)


	def _medium_membership(self: "MediumMembership", input_data: torch.Tensor):
		output = torch.Tensor(input_data.shape)
		for i in range(len(input_data)):
			for u in range(len(input_data[i])):
				x = input_data[i][u]
				curr_min = self.col_min[u]
				curr_max = self.col_max[u]

				#todo custom center
				center = (curr_max + curr_min) / 2

				if x <= curr_min:
					return 0
				elif curr_min < x and x <= center:
					return (x - curr_min) / (center - curr_min)
				elif center < x and x < curr_max:
					return 1 - ((x - center) / (curr_max - center))
				elif x >= curr_max:
					return 0
		return output

