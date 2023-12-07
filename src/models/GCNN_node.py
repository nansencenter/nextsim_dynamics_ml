import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNN_node(nn.Module):
    """
    Graph Convolutional Neural Network that takes a graph with node and edges features
    and outputs a 1D tensor.
    ...

    Arguments:

        num_features : int
            a formatted string to print out what the animal says
        hidden_channels : int
            Number of hidden channels in the conv layer
        output_size : int
            size of the 1D outpur tensor
    """

    def __init__(self, num_features, hidden_channels, output_size):
        super(GCNN_node, self).__init__()
        #2 conv layers as a test [WIP]
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, output_size)  

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index,edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index,edge_attr)
        x = F.relu(x)

        # Global pooling to aggregate node features (... not sure how elegant)
        x = torch.mean(x, dim=0)

        # Fully connected layer for the final output
        x = self.fc(x)
        
        
        return x


