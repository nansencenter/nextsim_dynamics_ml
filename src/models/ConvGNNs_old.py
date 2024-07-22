import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,ChebConv
from torch_geometric.nn.pool import global_mean_pool



class GCNN_node(nn.Module):

    def __init__(self, num_features, hidden_channels, output_size,dropout=0):
        super(GCNN_node, self).__init__()
        # conv layers as a test [WIP]
        self.conv1 = GCNConv(num_features, hidden_channels,K=1,dropout=dropout)
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

    def __init__(self, num_features1,num_features2, hidden_channels, output_size,dropout=0.1):
        super(GCNN_2G, self).__init__()
        # conv layers as a test [WIP]

        channels_e, channels_v = hidden_channels

        if len(channels_v) != len(channels_e):
            raise ValueError("Different number of layers for vertices and elements")
        if len(channels_v) == 0:
            raise ValueError("At least one layer is needed")
        if channels_e[-1] != channels_v[-1]:
            raise ValueError("Last layer must have the same number of channels for vertices and elements")

        self.convs1 = nn.ModuleList([ChebConv(num_features1, channels_e[0],K=1,dropout=dropout)])
        for i in range(len(channels_e)-1):
            self.convs1.append(ChebConv(channels_e[i], channels_e[i+1],K=1,dropout=dropout))


        self.convs2 = nn.ModuleList([ChebConv(num_features2, channels_v[0],K=1,dropout=dropout)])
        for i in range(len(channels_v)-1):
            self.convs2.append(ChebConv(channels_v[i], channels_v[i+1],K=1,dropout=dropout))


        self.fc = nn.Linear(channels_v[-1], output_size)

    def forward(self,g1,g2):
        x1 = g1.x
        x2 = g2.x

        for conv in self.convs1:
            x1 = conv(x1, g1.edge_index,g1.edge_attr)
            x1 = F.relu(x1)

        for conv in self.convs2:
            x2 = conv(x2, g2.edge_index,g2.edge_attr)
            x2 = F.relu(x2)


        x2 = global_mean_pool(x2, g2.batch)# Shape: (n_node(across al batches),features) -> (batch size,features)
        x1 = global_mean_pool(x1, g1.batch)      
     

        #concatenate both outputs
        x = torch.stack([x1,x2],dim=0) #Shape: (2,batch size,features)
    
        x = torch.mean(x,dim=0) #Shape: (batch size,features)
      
        # Fully connected layer for the final output
        x = self.fc(x)
       
        
        return x