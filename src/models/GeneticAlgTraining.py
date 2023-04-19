import numpy as np
import torch
import torch.nn as nn
import pygad
import pygad.torchga as ga
from torch.utils.data import DataLoader

from utils.data_loading import get_mnist_dataset
from utils.plotting import progress_bar

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
	'num_generations': 20,
	'num_parents_mating': 4,
	'parent_selection_type': 'tournament',
	'keep_parents': 1,
	'crossover_type': 'single_point',
	'mutation_type': 'random',
	'mutation_percent_genes': 10
}

class GeneticAlgTraining:
	def __init__(
		self: "GeneticAlgTraining", 
		model: nn.Module,
		config: GaConfig = defaultConfig,
		load_data: Callable[[], Tuple[torch.Tensor, torch.Tensor]] = get_mnist_dataset, 
	) -> None:

		self.loss_func = nn.MSELoss()
		self.config = config
		self.curr_iteration = 0

		# initalize model
		self.model = model
		#self.model = BinaryAutoEncoder(in_features, hidden_sizes, fuzzy_operators_per_layer)

		# load data
		train_data, _ = load_data()
		self.data_inputs = train_data.view(-1, 784)

	def fitness(self: "GeneticAlgTraining", ga_instance: pygad.GA, solution: List[float], solution_idx: int):
		# load weights into model
		models_weights_dict = ga.model_weights_as_dict(self.model, solution)
		self.model.load_state_dict(models_weights_dict)

		# forward pass through the net
		prediction = self.model(torch.Tensor(self.data_inputs))

		# calculate fitness
		fitness = 1 / (self.loss_func(prediction, self.data_inputs) + 0.0000001)
		return fitness.item()

	def new_gen_callback(self: "GeneticAlgTraining", ga_instance: pygad.GA):
		self.curr_iteration += 1
		progress_bar(self.curr_iteration, self.config['num_generations'])

	def train(self: "GeneticAlgTraining"):
		# no need for pytorch gradient evaluation
		self.model.eval()
		with torch.no_grad():
			torch_ga = ga.TorchGA(self.model, list(self.model.parameters())[0].shape[1])
			ga_instance = pygad.GA(
				initial_population=torch_ga.population_weights,
				num_generations=self.config['num_generations'],
				num_parents_mating=self.config['num_parents_mating'],
				fitness_func=self.fitness,
				#init_range_low=0,
				#init_range_high=1,
				parent_selection_type=self.config['parent_selection_type'],
				keep_parents=self.config['keep_parents'],
				crossover_type=self.config['crossover_type'],
				mutation_type=self.config['mutation_type'],
				#gene_type=int,
				#gene_space=[0,1],
				mutation_percent_genes=self.config['mutation_percent_genes'],
				on_generation=self.new_gen_callback
			)
			print("genetic algorithm intialization completed")

			ga_instance.run()
			solution, solution_fitness, solution_idx = ga_instance.best_solution()
			print("Parameters of the best solution : {solution}".format(solution=solution))
			print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

		models_weights_dict = ga.model_weights_as_dict(self.model, solution)
		self.model.load_state_dict(models_weights_dict)

		return self.model
	

