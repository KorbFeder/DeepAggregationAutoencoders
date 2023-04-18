import numpy as np
import pygad
import torch
from pygad import torchga

from models.AutoEncoder import AutoEncoder
from typing import Callable, List, Tuple, TypedDict

class GaConfig(TypedDict):
	num_generations: int
	num_parents_mating: int
	parent_selection_type: str
	keep_parents: int
	crossover_type: str
	mutation_type: str
	mutation_percent_genes: str

# default config if there is not config specified
defaultConfig: GaConfig = {
	'num_generations': 10000,
	'num_parents_mating': 4,
	'parent_selection_type': 'sss',
	'keep_parents': 1,
	'crossover_type': 'single_point',
	'mutation_type': 'random',
	'mutation_percent_genes': 10
}

model = AutoEncoder(8, [6, 4, 2, 4, 6])
original_data = np.array(
	[
		[
			1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0
		],
		[
			1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
		],
		[
			0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0
		]
	])

current_loader = iter(original_data)
data_inputs = next(current_loader)

def fitness(ga_instance: pygad.GA, solution: List[float], solution_idx: int):
	# load weights into model
	models_weights_dict = torchga.model_weights_as_dict(model, solution)
	model.load_state_dict(models_weights_dict)

	# forward pass through the net
	prediction = model(torch.Tensor(data_inputs))

	# calculate fitness
	fitness = 1 / (torch.nn.MSELoss()(prediction, torch.Tensor(data_inputs)) + 0.0000001)
	return fitness.item()


def new_gen_callback(ga_instance: pygad.GA):
	global current_loader, data_inputs
	try:
		best_solution, best_solution_fitness, best_match_idx = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)
		data_inputs = next(current_loader)
	except Exception as e:
		current_loader = iter(original_data)
		data_inputs = next(current_loader)
	

def test():
	config = defaultConfig

	torch_ga =torchga.TorchGA(model, list(model.parameters())[0].shape[1])
	ga_instance = pygad.GA(
		initial_population=torch_ga.population_weights,
		num_generations=config['num_generations'],
		num_parents_mating=config['num_parents_mating'],
		fitness_func=fitness,
		init_range_low=0,
		init_range_high=1,
		parent_selection_type=config['parent_selection_type'],
		keep_parents=config['keep_parents'],
		crossover_type=config['crossover_type'],
		mutation_type=config['mutation_type'],
		#gene_type=int,
		#gene_space=[0,1],
		mutation_percent_genes=config['mutation_percent_genes'],
		on_generation=new_gen_callback
	)
	ga_instance.run()
	solution, solution_fitness, solution_idx = ga_instance.best_solution()
	print("Parameters of the best solution : {solution}".format(solution=solution))
	print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
	
	a = next(current_loader)
	b = model(torch.Tensor(a))
	print(f"input is {a} output is {b}")

