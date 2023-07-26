def print_operators(autoencoder):
	for layer in autoencoder.layers:
		print("------------------")
		print([operator.name for operator in layer.operators])
		print("------------------")