import torch
from pygad import torchga
import pygad
import numpy

from models.AutoEncoder import AutoEncoder
from models.BinaryAutoEncoder import BinaryAutoEncoder
from utils.data_loading import get_mnist_dataloaders

def fitness_func(ga_instance, solution, sol_idx):
	global data_inputs, data_outputs, torch_ga, model, loss_function

	predictions = pygad.torchga.predict(model=model,
										solution=solution,
										data=data_inputs)

	solution_fitness = 1.0 / (loss_function(predictions, data_inputs).detach().numpy() + 0.00000001)

	return solution_fitness

def callback_generation(ga_instance):
	print("Generation = {generation}".format(generation=ga_instance.generations_completed))
	print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))

model = BinaryAutoEncoder(8, [4], [torch.max, torch.max, torch.max, torch.max])

torch_ga = torchga.TorchGA(model=model,
						num_solutions=10)

loss_function = torch.nn.MSELoss()

train_loader, test_loader = get_mnist_dataloaders()
train_loader = iter(train_loader)
#data_inputs = next(train_loader)[0].view(-1, 784)
data_inputs = torch.Tensor([1, 0, 1, 0, 1, 0, 1, 0])



def test():
	num_generations = 200 # Number of generations.
	num_parents_mating = 5 # Number of solutions to be selected as parents in the mating pool.
	initial_population = torch_ga.population_weights # Initial population of network weights.

	ga_instance = pygad.GA(
		num_generations=num_generations,
		num_parents_mating=num_parents_mating,
		initial_population=initial_population,
		gene_type=int,
		gene_space=[0,1],
		fitness_func=fitness_func,
		on_generation=callback_generation
	)

	ga_instance.run()

	ga_instance.plot_fitness(title="PyGAD & PyTorch - Iteration vs. Fitness", linewidth=4)

	solution, solution_fitness, solution_idx = ga_instance.best_solution()
	print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
	print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

	best_solution_weights = torchga.model_weights_as_dict(model=model,
															weights_vector=solution)
	model.load_state_dict(best_solution_weights)
	predictions = model(data_inputs)

	print(f"prediction of {data_inputs} is {predictions}, weights: {best_solution_weights}")


