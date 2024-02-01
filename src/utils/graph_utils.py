import numpy as np
import torch
from torch_geometric.data import Data
import torchvision.transforms as T


def standardize_graph_features(graph_list: list[Data]):
    """
    Standardize the features of the graphs in the graph_list.

    Arguments:
        graph_list: list[Data]
            A list of graphs.
    return:
        A list of graphs with standardized features.
    """
    all_features_tensor = torch.stack([graph.x for graph in graph_list])
    epsilon = 1e-10 #avoid /0
    std_per_channel = all_features_tensor.std(dim=[0,1]) + epsilon
    mean_per_channel = all_features_tensor.mean(dim=[0,1]) + epsilon
    transfrom = T.Normalize(mean=mean_per_channel,std=std_per_channel)
    normalized_features = transfrom(all_features_tensor.moveaxis(-1,0)).moveaxis(0,-1)

    #insert back the normalized features into the graphs
    for i,graph in enumerate(graph_list):
        graph.x = normalized_features[i,:,:]

    return graph_list