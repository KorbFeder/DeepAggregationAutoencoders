import torch
from model.edge_counting import EdgeCountingAutoencoder

torch.set_printoptions(precision=6, sci_mode=False)

def print_edge_counts(model: EdgeCountingAutoencoder):
    for layer in model.layers:
        print(layer.edge_type_count)