import torch
from fuzzy_logic.membership import Membership

class MediumMembership(Membership):
	def fuzzification(self: "MediumMembership", data: torch.Tensor) -> torch.Tensor:
		self.col_min = data.min(dim=0).values
		self.col_max = data.max(dim=0).values
		return self._medium_membership(data)	

	def defuzzification(self: "MediumMembership", input_data: torch.Tensor) -> torch.Tensor:
		output = torch.Tensor(input_data.shape)
		for i in range(len(input_data)):
			for u in range(len(input_data[i])):
				x = input_data[i][u]
				curr_min = self.col_min[u]
				curr_max = self.col_max[u]

				#todo custom center
				center = (curr_max + curr_min) / 2

				if x <= 0:
					return curr_min
				elif curr_min < x and x <= center:
					return x * (center - curr_min) + curr_min
				elif center < x and x < curr_max:
					return (1 - x) * (curr_max - center) + center
				elif x >= 1:
					return curr_min
		return output



	def _medium_membership(self: "MediumMembership", input_data: torch.Tensor) -> torch.Tensor:
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

