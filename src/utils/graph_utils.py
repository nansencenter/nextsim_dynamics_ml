import numpy as np
import torch
from torch_geometric.data import Data
import torchvision.transforms as T


def standardize_graph(graph_list: list[Data],normalize_targets:bool = True):
    """
    Standardize the features of the graphs in the graph_list.

    Arguments:
        graph_list: list[Data]
            A list of graphs.
        normalize_targets: bool
            If True, the targets will be normalized as well.
    return:
        graph_list: list[Data]
            A list of graphs with standardized features.
        inv_transform: NormalizeInverse
            The inverse transform to revert the normalization on coordinates.
    """
    #retrive all features as a single tensor
    all_features_tensor = torch.stack([graph.x for graph in graph_list])
    epsilon = 1e-10 #avoid /0
    #compute mean and std per channel
    std_per_channel_fea = all_features_tensor.std(dim=[0,1]) + epsilon
    mean_per_channel_fea = all_features_tensor.mean(dim=[0,1])
    #normalize features
    transfrom = T.Normalize(mean=mean_per_channel_fea,std=std_per_channel_fea)
    normalized_features = transfrom(all_features_tensor.moveaxis(-1,0)).moveaxis(0,-1)
    
    
    #normalize targets
    if normalize_targets:
        #retrive all targets as a single tensor
        all_targets_tensor = torch.stack([graph.y[0] for graph in graph_list])
        #reshape the tensor to have the shape (2(x,y),n_graphs,2) x,y work as channels in this case
        all_targets_tensor = all_targets_tensor.reshape(-1,2,2).moveaxis(1,0)
        #fetch mean and std of the input coords
        mean_coords, std_coords = mean_per_channel_fea[-2:], std_per_channel_fea[-2:]
        #normalize targets
        transfrom = T.Normalize(mean=mean_per_channel_fea[-2:],std=std_per_channel_fea[-2:])
        normalized_targets = transfrom(all_targets_tensor).moveaxis(0,1).reshape(-1,4)
        #create the inverse transform
        inv_transform = NormalizeInverse(mean=mean_coords,std=std_coords)
    else:
        inv_transform = None

    #insert back the normalized data into the graphs
    for i,graph in enumerate(graph_list):
        graph.x = normalized_features[i]
        if normalize_targets:
            graph.y[0] = normalized_targets[i]

    #compute inverse normalization
    
    
    return graph_list, inv_transform


class NormalizeInverse(T.Normalize):
    """
    Undoes the normalization and returns the reconstructed data in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())