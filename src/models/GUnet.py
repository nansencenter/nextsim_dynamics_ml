from torch_geometric.nn import GraphUNet
import torch
import torch.nn.functional as F

class GUNet(torch.nn.Module):
    def __init__(self, num_node_features, num_classes,depth=3, pool_ratios=0.5):
        super(GUNet, self).__init__()
        # Define the GraphUNet
        # The number of input channels should match the number of features each node has
        # In the case of Cora, each node has 1433 features
        # depth of 3, and each level reduces the nodes by a ratio of 1/2 using TopKPooling
        self.unet = GraphUNet(in_channels=num_node_features,
                              hidden_channels=32,
                              out_channels=num_classes,
                              depth=depth, 
                              pool_ratios=pool_ratios, 
                              sum_res=True)  # sum residual connections

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.unet(x, edge_index)
        return x