# DeepAggregationAutoencoders

The implementation of the Deep Aggregation Autoecoder learning algorithm.

The models used in the thesis are in
1. src/model/edge_counting.py:
   The implementation of the learning algorithm, where every layer has the same operators for all its nodes, fastest implementation.
3. src/model/daa.py:
   The implementation where the way the operators are used is chosable, meaning having alternating operators or random operators per neuron. 
5. src/model/autoencoder.py:
   The implementation of the a Autoencoder with a Artificial Neural Network used for comparison.
