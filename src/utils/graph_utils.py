import numpy as np
import torch

from tqdm import tqdm
from torch_geometric.data import Data


def get_trajectories(
    element_graphs:list[dict],
    element_index:int,
    iter=None
    ):
    """
    Given a list of element graphs, return the trajectories of the nodes that compose the element
    
    Arguments:
        element_graphs : list
            List of element graphs
        element_index : int
            index to of element to track
        iter: int
            iterations to track, if None, compute all trajectory
    returns:
        trajectories : torch.tensor"""
    
    #Get target element's nodes
    nodes = element_graphs[0]['t'][element_index]
    #Get nodes indeces
    node_idx = element_graphs[0]['i'][nodes]

    x,y = [],[]
    for i,hour in enumerate(element_graphs):
        if iter is None or i<iter:
            #having 3 separate index is necesary to keep track of each in x,y
            idx_1 = np.where(hour['i']==node_idx[0])[0]
            idx_2 = np.where(hour['i']==node_idx[1])[0]
            idx_3 = np.where(hour['i']==node_idx[2])[0]
            if len(idx_1)>0 and len(idx_2)>0 and len(idx_3)>0:
                    x.append(hour['node_x'][[idx_1,idx_2,idx_3]])
                    y.append(hour['node_y'][[idx_1,idx_2,idx_3]])

        
    x = np.stack(x)
    y = np.stack(y)

    trajectories = torch.stack([torch.tensor(x),torch.tensor(y)],dim=0).squeeze()
    return trajectories




def interpolate_node_into_element(
        bin_files: list[dict],
):
    """
    Function to interpolate the element information into the neighbouring nodes.

    Arguments:
        bin_files: list[dict]
            list of dictionaries containing nextsim outputs        
    returns:
        element_graphs: list[dict]
            list of dictionaries containing the element based graphs
    """

    
    element_graphs = []

    for file in tqdm(bin_files,"Interpolating node info into elements"):
        #fetch all node features
        node_features = [feat for feat in file.keys() if file[feat].shape == file['i'].shape and feat != 'i']
        #create a element graph
        element_graph = {
            key:item for key,item in file.items() if key not in node_features and key != 'i'
        }
        #Average quantities
        for feature in node_features:
            element_graph[feature] = (file[feature][file['t']].sum(axis=-1)/3)
        
        #keep track of x,y
        element_graph['node_x'] = file['x']
        element_graph['node_y'] = file['y']

        element_graphs.append(element_graph)
       

    return element_graphs




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
        target_element: int
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
        target_coords = get_trajectories(element_graphs[i:],target_element,1)

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