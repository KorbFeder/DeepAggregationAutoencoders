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
		train_forward, normal_forward = layer.forward(torch.Tensor([[1, 2, 3, 4, 5, 6, 7, 8], [-1, -2, -3, -4, -5, -6, -7, -8]]), is_train=True)
		self.assertTrue(torch.equal(train_forward, torch.Tensor([[[1, 3, 6, 1, 2, 5], [3, 5, 8, 5, 4, 7]], [[-3, -5, -8, -5, -4, -7], [-1, -3, -6, -1, -2, -5]]])))
		self.assertTrue(torch.equal(normal_forward, torch.Tensor([[1, 5, 8, 1, 4, 5], [-3, -3, -6, -5, -2, -7]])))
	
	def test_autoencoder_forward(self: "DeepAggrAutoencoderTest"):
		self.assertEqual(0, 0)

if __name__ == '__main__':
	unittest.main()