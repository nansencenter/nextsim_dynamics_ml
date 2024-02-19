import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv
from torch_geometric.nn.pool import global_mean_pool



class GCNN_node(nn.Module):

    def __init__(self, num_features, hidden_channels, output_size,dropout=0):
        super(GCNN_node, self).__init__()
        # conv layers as a test [WIP]
        self.conv1 = ChebConv(num_features, hidden_channels,K=1,dropout=dropout)
        self.fc = nn.Linear(hidden_channels, output_size)

    def forward(self, x, edge_index,edge_attr):
        x = self.conv1(x, edge_index,edge_attr)
        x = F.relu(x)

        # Global pooling to aggregate node features (... not sure how elegant)
        x = torch.mean(x, dim=0)

        # Fully connected layer for the final output
        x = self.fc(x)


        return x
    


class GCNN_2G(nn.Module):

    def __init__(self, num_features1,num_features2, hidden_channels, output_size):
        super(GCNN_2G, self).__init__()
        # conv layers as a test [WIP]
        self.conv1 = ChebConv(num_features1, hidden_channels,K=1)
        self.conv2 = ChebConv(num_features2, hidden_channels,K=1)
        self.fc = nn.Linear(hidden_channels, output_size)

    def forward(self,g1,g2):
       
        x1 = self.conv1(g1.x, g1.edge_index,g1.edge_attr)
        x1 = F.relu(x1)

        x2 = self.conv2(g2.x, g2.edge_index,g2.edge_attr)
        x2 = F.relu(x2)


        x2 = global_mean_pool(x2, g2.batch)
        x1 = global_mean_pool(x1, g1.batch)      

        #concatenate both outputs
        x = torch.stack([x1,x2],dim=0)
        x = torch.mean(x,dim=0)
        # Fully connected layer for the final output
        x = self.fc(x)
       
        
        return x