from globals.data_set_name import DatasetName
from logger.plot_mnist_outputs import plot_mnist_outputs
from logger.plot_wine_results import plot_wine_results

def get_result_plotting(name: DatasetName):
	if name == DatasetName.mnist.value:
		return plot_mnist_outputs
	elif name == DatasetName.wine.value:
		#return plot_wine_results
		return None
	else:
		return None