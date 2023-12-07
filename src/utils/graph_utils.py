import numpy as np
import torch

from tqdm import tqdm
from torch_geometric.data import Data




def interpolate_element_into_nodes(
        bin_files: list[dict],
        features: list[tuple]
):
    """
    Function to interpolate the element information into the neighbouring nodes.
    

    Arguments:
        bin_files: list[dict]
            list of dictionaries containing nextsim outputs
        features: list[tuple]
            list of node and element feature names
        
    returns:
        bin_files: list[dict]
            list of dictionaries containing the interpolated information
    """

    for file in tqdm(bin_files,"Interpolating features from element to nodes..."):

        
        #initialize node variables to 0
        node_feature_shape = file['x'].shape
        for node_feature,_ in features:
            file[node_feature] = np.zeros(node_feature_shape)
        #sum_elements will keep track of how many elements have contributed to the node
        file['sum_elements'] = np.zeros(node_feature_shape)

        #For every element add its content to the neighbouring nodes
        for i,element in enumerate(file['t']):

            for node_feature,element_feature in features:
                file[node_feature][element] += file[element_feature][i]            
            file['sum_elements'][element] += 1

        #Average by the number of contributing elements
        for node_feature,_ in features:
             file[node_feature] = file[node_feature]/file['sum_elements']
       

    return bin_files



def bin_to_torchGraph(
        bin_files: list[dict],
        features_list: list[str],
        target_idx: int
):

    """
    Funtion to create torch graph "Data" structures from nextsim bin output files (as .npz)   
    For each file 3 torch.tensors are computed and used to create the data object:
        node features [num_nodes, num_node_features]
        edges [2, num_edges]
        edges atributtes [num_edges, num_edge_features]

    Arguments:
        bin_files: list[dict]
            list of dictionaries containing nextsim outputs
        features: list[str]
            list of features to include on each node
        target_idx: int
            index of the target node
        
    returns:
        grads : list[torch_geometric.data.Data]
            list containing torch geometric data objetcs
            
    """
    graph_list=[]

    for idx,hour_graph in tqdm(enumerate(bin_files),"Converting bins to torch graphs..."):

        #get the next pos of target node
        target_coords = torch.tensor([bin_files[idx+1]['x'][target_idx],bin_files[idx+1]['y'][target_idx]])

        #concat all node features sequentially (following index(t) number) in a tensor
        features = []
        idx_list = [] #keep track of index inside node feature tensor
        for key,item in hour_graph.items():

            if key in features_list:
                idx_list.append(key)
                features.append(torch.tensor(np.array([item])))

        if len(features)>0:
            node_features = torch.cat(features).t().to(torch.float32)
        else:
            raise ValueError("None of the features specified is on the graph")

        #find all distinct (undirected) edges from every triangle
        edges = np.concatenate([
            hour_graph['t'].transpose()[:2],
            hour_graph['t'].transpose()[1:],
            hour_graph['t'].transpose()[0:-1]
        ],axis=-1)
        edges = torch.tensor(np.unique(edges,axis=-1))

        #Now we need to consult x,y coordinates of each node of the edges and compute the edge distance
        # for each each row of edge ends we retrieve this info by index
        # and we stack it as a 2xE (2 for each edge end, E as number of edges)
        coord_idx= [i for i,key in enumerate(idx_list) if key in ['x','y']]
        if len(coord_idx)==2:
            edges_coordinates = [
                torch.stack(
                    [
                        node_features[edge_row][:,coord_idx[0]],
                        node_features[edge_row][:,coord_idx[1]]
                    ]
                )
                for edge_row in edges
            ]
        else:
            raise ValueError("Unable to find coordinates for nodes in graph mesh. \nDid you include it in the feature list?")
        
            
        #now we can compute the norm of each edge vector using torch api
        # unsqueeze to match [num_edges, num_edge_features] shape
        edge_attr = torch.norm(edges_coordinates[1] - edges_coordinates[0],dim=0).unsqueeze(dim=-1).to(torch.float32)

        #Now we can create our torch-geometric graph using the "Data" class
        ice_graph = Data(x=node_features, edge_index=edges, edge_attr=edge_attr)
        
        graph_list.append(ice_graph)

        return graph_list