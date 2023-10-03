import torch
from archived_models.node_counting import NodeCountingAutoencoder

torch.set_printoptions(precision=6, sci_mode=False)

def print_node_counts(model: NodeCountingAutoencoder):
    for layer in model.layers:
        print(layer.node_type_count)