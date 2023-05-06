import unittest
import torch

from model.deep_aggr_autoencoder import DeepAggregateLayer, DeepAggregateAutoEncoder, create_operator_table

class DeepAggrAutoencoderTest(unittest.TestCase):
	def test_layer_forward(self: "DeepAggrAutoencoderTest"):
		layer = DeepAggregateLayer(8, 6, 3, create_operator_table())
		layer.operator_table_indices = [0, 1, 1, 0, 1, 0]
		layer.connection_indices = torch.Tensor([[0, 1, 2], [2, 3, 4], [5, 6, 7], [0, 1, 4], [1, 2, 3], [4, 5, 6]])
		result = layer.forward(torch.Tensor([[1, 2, 3, 4, 5, 6, 7, 8], [-1, -2, -3, -4, -5, -6, -7, -8]]))
		self.assertTrue(torch.equal(result, torch.Tensor([[1, 5, 8, 1, 4, 5], [-3, -3, -6, -5, -2, -7]])))

	def test_layer_forward_train(self: "DeepAggrAutoencoderTest"):
		layer = DeepAggregateLayer(8, 6, 3, create_operator_table())
		layer.operator_table_indices = [0, 1, 1, 0, 1, 0]
		layer.connection_indices = torch.Tensor([[0, 1, 2], [2, 3, 4], [5, 6, 7], [0, 1, 4], [1, 2, 3], [4, 5, 6]])
		normal_forward, train_forward = layer.forward(torch.Tensor([[1, 2, 3, 4, 5, 6, 7, 8], [-1, -2, -3, -4, -5, -6, -7, -8]]), is_train=True)
		self.assertTrue(torch.equal(train_forward, torch.Tensor([[[1, 3, 6, 1, 2, 5], [3, 5, 8, 5, 4, 7]], [[-3, -5, -8, -5, -4, -7], [-1, -3, -6, -1, -2, -5]]])))
		self.assertTrue(torch.equal(normal_forward, torch.Tensor([[1, 5, 8, 1, 4, 5], [-3, -3, -6, -5, -2, -7]])))
	
	def test_autoencoder_forward(self: "DeepAggrAutoencoderTest"):
		daae = DeepAggregateAutoEncoder(8, [6], [3, 3])
		daae.layers = [
			DeepAggregateLayer(8, 6, 3, create_operator_table()), 
		]
		daae.output_layer = DeepAggregateLayer(6, 8, 3, create_operator_table())

		daae.layers[0].operator_table_indices = [0, 1, 1, 0, 1, 0]
		daae.output_layer.operator_table_indices = [0, 1, 1, 0, 1, 0, 1, 1]
		daae.layers[0].connection_indices = torch.Tensor([[0, 1, 2], [2, 3, 4], [5, 6, 7], [0, 1, 4], [1, 2, 3], [4, 5, 6]])
		daae.output_layer.connection_indices = torch.Tensor([
			[0, 1, 2], [2, 3, 4], [5, 0, 1], [2, 3, 4], [5, 2, 3], [3, 4, 5], [2, 4, 5], [0, 2, 4]
		])

		prediciton = daae.forward(torch.Tensor([[1, 2, 3, 4, 5, 6, 7, 8], [-1, -2, -3, -4, -5, -6, -7, -8]]))
		y = torch.Tensor([[1, 8, 5, 1, 8, 1, 8, 8], [-6, -2, -3, -6, -5, -7, -2, -2]])
		self.assertTrue(torch.equal(prediciton, y))
	
	def test_autoencoder_forward_train(self: "DeepAggrAutoencoderTest"):
		daae = DeepAggregateAutoEncoder(8, [6], [3, 3])
		daae.layers = [
			DeepAggregateLayer(8, 6, 3, create_operator_table()), 
		]
		daae.output_layer = DeepAggregateLayer(6, 8, 3, create_operator_table())

		daae.layers[0].operator_table_indices = [0, 1, 1, 0, 1, 0]
		daae.output_layer.operator_table_indices = [0, 1, 1, 0, 1, 0, 1, 1]
		daae.layers[0].connection_indices = torch.Tensor([[0, 1, 2], [2, 3, 4], [5, 6, 7], [0, 1, 4], [1, 2, 3], [4, 5, 6]])
		daae.output_layer.connection_indices = torch.Tensor([
			[0, 1, 2], [2, 3, 4], [5, 0, 1], [2, 3, 4], [5, 2, 3], [3, 4, 5], [2, 4, 5], [0, 2, 4]
		])

		prediciton, activation = daae.forward(torch.Tensor([[1, 2, 3, 4, 5, 6, 7, 8], [-1, -2, -3, -4, -5, -6, -7, -8]]), is_train=True)
		y = torch.Tensor([[1, 8, 5, 1, 8, 1, 8, 8], [-6, -2, -3, -6, -5, -7, -2, -2]])
		y_activation = torch.Tensor([
			[
				[1, 3, 6, 1, 2, 5], 
				[3, 5, 8, 5, 4, 7], 
			], [
				[-3, -5, -8, -5, -4, -7],
				[-1, -3, -6, -1, -2, -5]
			]
		])

		self.assertTrue(torch.equal(prediciton, y))
		self.assertTrue(torch.equal(y_activation, activation))
	

if __name__ == '__main__':
	unittest.main()