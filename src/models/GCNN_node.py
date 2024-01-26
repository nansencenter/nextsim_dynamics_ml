import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv


class GCNN_node(nn.Module):

    def __init__(self, num_features, hidden_channels, output_size):
        super(GCNN_node, self).__init__()
        # conv layers as a test [WIP]
        self.conv1 = ChebConv(num_features, hidden_channels,K=1)
        self.fc = nn.Linear(hidden_channels, output_size)

    def forward(self, x, edge_index,edge_attr):
        x = self.conv1(x, edge_index,edge_attr)
        x = F.relu(x)

        # Global pooling to aggregate node features (... not sure how elegant)
        x = torch.mean(x, dim=0)

        # Fully connected layer for the final output
        x = self.fc(x)


        return x