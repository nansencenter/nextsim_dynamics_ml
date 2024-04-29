import numpy as np
import torch 
from torch_geometric.data import Data
from tqdm import tqdm

from utils.Tri_neighbors import TriNeighbors
from scipy.interpolate import LinearNDInterpolator



class IceData(Data):
   """
   Overwrite the torch geometric.Data class to adapt data for mini-batch processing
   """
   def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'edge_index' or key =='pos':  #cat along dim 1
            return 1
        elif key == 'x': 
            return 0 #cat along dim 0
        elif key == 'y' or key == 'metadata':
            return None # create a new batch dimension
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)

class Nextsim_data():
    """
    Class to manage data from nextsim outputs.
    """
    
    def __init__(self,file_graphs,vertex_element_features: list[str] = ['x','y'],d_time:int = 3600, step = 1) -> None:
     
        self.file_graphs = file_graphs
        self.d_time = d_time
        self.step = step
        #get vertex and element data
        self.vertex_data_list = self.compute_vertex_data()
        self.element_data_list = self.compute_element_data(vertex_element_features)

        self.tri_neighbors = {}
        self.forcings = {}

    def get_num_elements(self,time_index:int = 0):
        return len(self.element_data_list[time_index]['x'])
    

    def get_num_vertexs(self,time_index:int = 0):
        return len(self.vertex_data_list[time_index]['x'])
    

    def get_tri_neighbors(self,time_index:int = 0):

        if time_index not in self.tri_neighbors.keys():
            self.tri_neighbors[time_index] = TriNeighbors(self.element_data_list[time_index]['t'])

        return self.tri_neighbors[time_index]
    
    def get_forcings(self,time_index:int = 0,features: list[str] = ['M_wind_x', 'M_wind_y', 'M_ocean_x', 'M_ocean_y']):
        """
        Function to get the interpolated forcing fields at a given time index.

        Arguments:
            time_index: int
                index of the time to sample from
            features: list[str]
                list of features to interpolate

        returns:
            forcings: dict
                dictionary of forcing fields

        """
        
        if time_index not in self.forcings.keys() or not np.isin(features, self.forcings[time_index].keys()).all():
            self.forcings[time_index] = {}
            for feature in features:
                if feature in self.vertex_data_list[time_index].keys():
                    field = self.vertex_data_list[time_index]
                else:
                    field = self.element_data_list[time_index]

                self.forcings[time_index][feature] = LinearNDInterpolator(list(zip(field['x'], field['y'])), field[feature])
        
        return self.forcings[time_index]

    def get_item(self, time_index:int = 0, elements:bool = True):
        """
        Function to get data at a given time index.

        Arguments:
            time_index: int
                index of the time to sample from
            elements: bool
                if True, sample from elements, else sample from vertexs
        returns:    
            data: dict
                dictionary of data
        """
        if elements:
            data = self.element_data_list[time_index]
        else:
            data = self.vertex_data_list[time_index]

        return data


    def compute_element_data(self,vertex_element_features: list[str] = ['x','y']):
        """
        Function to fetch element information and interpolate vertex features.

        Arguments:
            vertex_features: list[str]    
                list of vertex feautres to interpolate into the elements        
        returns:
            element_data: list[dict]
                list of dictionaries containing the element based graphs
        """

        
        element_data_list = []

        for file in self.file_graphs:
            #create a element graph
            element_data = {
                key:item for key,item in file.items() if key not in vertex_element_features and item.shape != file['i'].shape
            }
            #Average quantities
            for feature in vertex_element_features:
                element_data[feature] = (file[feature][file['t']].sum(axis=-1)/3)
            
            #keep track of x,y
            element_data['vertex_x'] = file['x']
            element_data['vertex_y'] = file['y']
            element_data['i'] = file['i']

            element_data_list.append(element_data)
           

        return element_data_list
    
    def compute_vertex_data(self):
        """
        Function to fetch vertex information.
        """
        vertex_data_list = []
        for file in self.file_graphs:

            #fetch all vertex features
            vertex_data = {
                key:item for key,item in file.items() if item.shape == file['i'].shape or key == 't'
            }

            vertex_data_list.append(vertex_data)           
        

        return vertex_data_list

    def compute_velocity(self,time_index:int):
        """
        Function to compute the velocity.

        Arguments:
            time_index: int
                index of the time to sample from
        """

        d0 = self.vertex_data_list[time_index-1]
        d1 = self.vertex_data_list[time_index]

        #compute vel between d0 and d1
        _,comm01, comm10 = np.intersect1d(d0['i'], d1['i'], assume_unique=True, return_indices=True)
        # get common X, Y coordinates
        x0 = d0['x'][comm01]
        y0 = d0['y'][comm01]
        x1 = d1['x'][comm10]
        y1 = d1['y'][comm10]

        # compute drift [m/s] for two files separated by delta time
        u = (x1 - x0) / (self.d_time)
        v = (y1 - y0) / (self.d_time)
        if len(v) != len(d0['x']):
            u_interp = LinearNDInterpolator(list(zip(x0, y0)), u)
            v_interp = LinearNDInterpolator(list(zip(x0, y0)), v)

            v = v_interp( d0['x'],  d0['y'])
            u = u_interp( d0['x'],  d0['y'])

        return torch.stack([torch.tensor(u),torch.tensor(v)],dim=1)
    

    def compute_acceleration(self,time_index:int):
        """
        Function to compute the acceleration .

        Arguments:
            time_index: int
                index of the time to sample from
        returns:    
            acceleration: torch.tensor
                tensor of acceleration
        """
        d0 = self.vertex_data_list[time_index-1]
        d1 = self.vertex_data_list[time_index]
        d2 = self.vertex_data_list[time_index+1]

       #compute vel between d0 and d1
        _,comm01, comm10 = np.intersect1d(d0['i'], d1['i'], assume_unique=True, return_indices=True)
        # get common X, Y coordinates
        x0 = d0['x'][comm01]
        y0 = d0['y'][comm01]
        x1_0 = d1['x'][comm10]
        y1_0 = d1['y'][comm10]

        # compute drift [m/s] for two files separated by delta time
        u0 = (x1_0 - x0) / (self.d_time)
        v0 = (y1_0 - y0) / (self.d_time)


        #compute vel between d1 and d2
        _,comm12, comm21 = np.intersect1d(d1['i'], d2['i'], assume_unique=True, return_indices=True)
        # get common X, Y coordinates
        x1_1 = d1['x'][comm12]
        y1_1 = d1['y'][comm12]
        x2 = d2['x'][comm21]
        y2 = d2['y'][comm21]

        # compute drift [m/s] for two files separated by delta time
        u1 = (x2 - x1_1) / (self.d_time)
        v1 = (y2 - y1_1) / (self.d_time)

        if len(x0) != len(x2):
        
            u0_interp = LinearNDInterpolator(list(zip(x1_0, y1_0)), u0)
            v0_interp = LinearNDInterpolator(list(zip(x1_0, y1_0)), v0)

            u1_interp = LinearNDInterpolator(list(zip(x1_1, y1_1)), u1)
            v1_interp = LinearNDInterpolator(list(zip(x1_1, y1_1)), v1)

            v0 = v0_interp( d1['x'],  d1['y'])
            u0 = u0_interp( d1['x'],  d1['y'])

            v1 = v1_interp( d1['x'],  d1['y'])
            u1 = u1_interp( d1['x'],  d1['y'])


        #compute acceleration
        a = (u1 - u0) / (self.d_time)
        b = (v1 - v0) / (self.d_time)


        return torch.stack([torch.tensor(a),torch.tensor(b)],dim=1)

    def get_closer_neighbours(
            self,
            dataindex:int,
            n_neighbours: int,
            time_index:int = 0,
            elements:bool = True,
    ):
        """
        Function to get the closest elements/vertexs to a given index.

        Arguments:
            dataindex: int
                index of the element to get the closest neighbours from
            n_neighbours: int
                number of neighbours to return
            time_index: int
                index of the time to sample from
            elements: bool
                if True, sample from elements, else sample from vertexs
        returns:    
            neighbours: np.array
                array of neighbours
        """
        center_x,center_y = self.element_data_list[time_index]['x'][dataindex],self.element_data_list[time_index]['y'][dataindex]

        if elements:
            data_x,data_y = self.element_data_list[time_index]['x'],self.element_data_list[time_index]['y']
        else:
            data_x,data_y = self.vertex_data_list[time_index]['x'],self.vertex_data_list[time_index]['y']

        neighbours = np.argsort(np.sqrt((data_x-center_x)**2 + (data_y-center_y)**2))[:n_neighbours]

        return neighbours


    def compute_vertex_neighbourhood(
            self,
            vertex_index:int,
            time_index:int = 0,
            return_vertex:bool = False,
            n_neighbours:int = 1
    ):
        """ 
        Function to compute neighbourhood of a given vertex

        Arguments:

            vertex_i: int
                i index of the given element
            time index: int
                time index of the mesh
            return_vertex:bool
                if true return also vertexs in the neighbourhood
        returns
            elements: torch.tensor
                elements in the neighbourhood
            vertex: torch.tensor
                vertex in the neighbourhood

        """
        
        elements = np.where(self.element_data_list[time_index]['t']==[vertex_index])[0]
        adj_elements = elements #just adjacent elements (case neighbour = 1)

        if n_neighbours>1:
            elements = self.get_tri_neighbors(time_index).get_neighbors_many(elements,n_neighbours-1)

        vertexs = None
        if return_vertex:
            vertexs = np.unique(self.vertex_data_list[time_index]['t'][elements].flatten())
        
        return adj_elements,elements,vertexs
        



    def get_samples_area(
            self,
            central_coords: tuple[float,float],
            radius: float,
            time_index:int = 0,
            n_samples:int = None,
            elements:bool = True,
    ):
        """
        Function to get samples from a circular area.

        Arguments:
            central_coords: tuple[float,float]
                coordinates of the center of the circle
            radius: float
                radius of the circle
            time_index: int
                index of the time to sample from
            n_samples: int
                number of samples to return
            elements: bool
                if True, sample from elements, else sample from mesh vertex
        returns:    
            samples: np.array
                array of samples
        """
        
        if elements:
            data = self.element_data_list[time_index]
        else:
            data = self.vertex_data_list[time_index]

        x_center,y_center = central_coords
        #get the data that are inside the circle
        samples = np.where(np.sqrt((data['x']-x_center)**2 + (data['y']-y_center)**2)<radius)[0]
        if n_samples is not None:
            n_samples = min(n_samples,len(samples))
            samples = np.random.choice(samples,n_samples)

        samples_i = data['i'][samples]

        
        return samples_i,samples
        

    def get_vertex_trajectories(
            self,
            time_index:int,
            vertex_i:int,
            iter=2,
            velocity:bool = False
    ):
        """ Function that returns the trajectories of the vertexs that compose the element

        Arguments:
            time_index : int
                index to of time to track from
            vertex_index : int
                index to of vertex to track
            iter: int
                iterations to track, if None, compute all trajectory
        returns:
            trajectories : torch.tensor"""

        if velocity:
            iter += 1

        x,y = [],[]
        for mesh in self.vertex_data_list[time_index:time_index+iter]:

            idx = np.where(mesh['i']==vertex_i)[0]
            if len(idx)>0:
                x.append(mesh['x'][idx])
                y.append(mesh['y'][idx])

        x,y = torch.as_tensor(np.array(x)),torch.as_tensor(np.array(y)) 
        #return velocity instead of positions
        if velocity:   
            return torch.stack([x,y],dim=1).squeeze().diff(dim=0).div(self.d_time)
        else:
            return torch.stack([x,y],dim=1).squeeze()

        

    def get_element_trajectories(
            self,
            time_index:int,
            element_index:int,
            iter=None,
            elements:bool = True,
    ):
        """ Function that returns the trajectories of the vertexs that compose the element

        Arguments:
            time_index : int
                index to of time to track from
            element_index : int
                index to of element to track
            iter: int
                iterations to track, if None, compute all trajectory
        returns:
            trajectories : torch.tensor"""

        #Get target element's vertexs
        vertexes = self.element_data_list[time_index]['t'][element_index]
        #Get vertexs indeces
        vertex_i = self.element_data_list[time_index]['i'][vertexes]

        x,y = [],[]
        for i,hour in enumerate(self.element_data_list[time_index:]):
            if iter is None or i<iter:
                #we need to retrieve the position (idx) of each vertex by index i
                #having 3 separate index is necesary to keep track of each in x,y
                idx_1 = np.where(hour['i']==vertex_i[0])[0]
                idx_2 = np.where(hour['i']==vertex_i[1])[0]
                idx_3 = np.where(hour['i']==vertex_i[2])[0]
                if len(idx_1)>0 and len(idx_2)>0 and len(idx_3)>0:
                        x.append(hour['vertex_x'][[idx_1,idx_2,idx_3]])
                        y.append(hour['vertex_y'][[idx_1,idx_2,idx_3]])


        x = np.stack(x)
        y = np.stack(y)

        trajectories = torch.stack([torch.tensor(x),torch.tensor(y)],dim=0).squeeze()
        if elements:
            trajectories = trajectories.mean(dim=-1)
        return trajectories
    


class Ice_graph(Nextsim_data):
    """
    Class to create torch geometric graphs data from nextsim outputs.
    """
    def __init__(self,file_graphs,vertex_element_features: list[str] = ['x','y'],d_time:int = 3600, step = 1) -> None:
        super().__init__(file_graphs,vertex_element_features,d_time,step)
    

    def get_samples_graph(
            self,
            central_coords: tuple[float,float],
            radius: float,
            time_indeces:list[int]=None,
            n_samples:int = 1000,
            target_iter:int = 5,
            e_features: list[str] = ['Concentration', 'Thickness', 'M_wind_x', 'M_wind_y', 'M_ocean_x', 'M_ocean_y', 'x', 'y'],
            include_vertex:bool = False,
            pred_velocity:bool = False,
            n_neighbours = 1,
            future_forcing:bool = True
    ):
        """
        Function to get the graph of a given area at a given time index.

        Arguments:
            central_coords: tuple[float,float]
                coordinates of the center of the circle
            radius: float
                radius of the circle
            time_indeces: lies[int]
                indeces of the time to sample from
            n_samples: int
                number of samples to return per time index
            target_iter: int
                iteration to predict
            e_features: list[str]
                list of features to include in the element graph
            include_vertex: bool
                if True, include the vertex graph
            pred_velocity: bool
                if True, predict the velocity instead of the position           

        returns:
            graph: torch_geometric.Data
        """
        graph_list = []

        for t in time_indeces:
            samples = self.get_samples_area(central_coords,radius,t,n_samples,False)
            for sample in tqdm(samples, f"Generating samples at time {t}"):
                graph = self.get_vertex_centered_graph(vertex_i=sample,time_index=t,target_iter=target_iter,e_features=e_features,include_vertex=include_vertex,velocity=pred_velocity,n_neighbours=n_neighbours,future_forcing=future_forcing)
                if graph is not None:
                    graph_list.append(graph)

        return graph_list
    
    def get_vertex_centered_graph(
            self,
            vertex_i,
            time_index:int = 0,
            target_iter:int = 1,
            e_features: list[str] = ['Damage', 'Concentration', 'Thickness', 'M_wind_x', 'M_wind_y', 'M_ocean_x', 'M_ocean_y', 'x', 'y'],
            v_features: list[str] = ['M_VT_x','M_VT_y','M_wind_x', 'M_wind_y', 'M_ocean_x', 'M_ocean_y', 'x', 'y'],
            include_vertex:bool = False,
            velocity:bool = False,
            n_neighbours:int = 1,
            future_forcing:bool = True
    ):
        """
        Function to get the graph of elements around a given vertex at a given time index.

        Arguments:
            vertex_index: int
                i of the vertex to sample from
            time_index: int
                index of the time to sample from
            target_iter: int
                iteration to predict
        returns:
            graph: torch_geometric.data.Data

        """

        element_data = self.get_item(time_index,elements=True)
        vertex_data = self.get_item(time_index,elements=False)

        #get vertex index
        vertex_index = np.where(vertex_data['i']==vertex_i)[0][0]
        #get the neighbours
        adj_elements,e_neighbours,v_neighbours = self.compute_vertex_neighbourhood(vertex_index=vertex_index,time_index=time_index,return_vertex=include_vertex,n_neighbours=n_neighbours)

        #skip when there is no ice in adjacents elements 
        if element_data['Concentration'][adj_elements].min() == 0:
            return None
        #get target coordinates
        target = self.get_vertex_trajectories(time_index,vertex_i=vertex_i,iter= self.step,velocity=velocity).squeeze().to(torch.float32)

        #Step
        if  self.step>1 and len(target.size())>0 and target.shape[0] == self.step:
            target = target.mean(dim=0)
        elif self.step==1:
            pass
        else:
            return None

        if len(target.size())==0 or target.shape[0] != 2:
            return None    #skip vertexs / elements that disapear

        if not velocity:
            target = target.flatten()


        #store initial coordinates for visulisazion
        vertex_idx = vertex_index
        x_center = vertex_data['x'][vertex_idx]
        y_center = vertex_data['y'][vertex_idx]
        element_coords = torch.tensor(np.array([x_center,y_center]))
        y = [target,element_coords]

        #get node features
        if future_forcing:
            node_features, features_indeces = self.__get_node_features(element_data,e_features,e_neighbours,time_index)
        else:
            node_features, features_indeces = self.__get_node_features(element_data,e_features,e_neighbours)

        #get edge features
        edge_index = self.__get_element_edge_index(element_data,e_neighbours,edge_conectivity=True)

        #get edge distances and node positions
        edge_dist,positions = self.__compute_edge_distances(feature_indeces=features_indeces,node_features=node_features,edge_index=edge_index)

        #exclude xy
        idx_no_xy = np.where(np.isin(features_indeces,['x','y'],invert=True))[0]
        features_indeces = features_indeces[idx_no_xy]
        node_features = node_features[:,idx_no_xy]

        metadata = {
            "time": time_index,
            "vertex_i": vertex_data['i'][v_neighbours]
        }
        #Now we can create our torch-geometric graph using the "Data" class
        e_graph = IceData(x=node_features, edge_index=edge_index,edge_attr=edge_dist,pos=positions,y=y, metadata=metadata)
        v_graph = None

        if include_vertex:
            #get node features
            if future_forcing:
                node_features, features_indeces = self.__get_node_features(vertex_data,v_features,v_neighbours,time_index)
            else:
                node_features, features_indeces = self.__get_node_features(vertex_data,v_features,v_neighbours)
                
            #get edge features
            edge_index = self.get_vertex_edge_index(vertex_neighbours=v_neighbours,triangles=element_data['t'][e_neighbours])
            #get edge distances and node positions
            edge_dist,positions = self.__compute_edge_distances(feature_indeces=features_indeces,node_features=node_features,edge_index=edge_index)

            #exclude xy
            idx_no_xy = np.where(np.isin(features_indeces,['x','y'],invert=True))[0]
            features_indeces = features_indeces[idx_no_xy]
            node_features = node_features[:,idx_no_xy]
           
            #Now we can create our torch-geometric graph using the "Data" class
            v_graph = IceData(x=node_features, edge_index=edge_index, edge_attr=edge_dist,pos=positions,y=y,metadata=metadata)

        return e_graph, v_graph

    def __get_node_features(
            self,
            field: dict,
            features: list[str],
            indexes: np.array,
            time_index:int = None
    ):
        """
        Function to get the node features of a given field.

        Arguments:
            field: dict
                dictionary of data
            indexes: np.array
                indexes of the elements to sample from
            features: list[str]
                list of features 
            time_index: int
                index of the time to sample from
        returns:
            features: torch.tensor
                tensor of features
            idx_list: list
                list of features names
        """
        
        #concat all node features sequentially (following index(t)) in a tensor
        node_features = []
        idx_list = []
        for key,item in field.items():
            if key in features:
                idx_list.append(key)
                node_features.append(torch.tensor(np.array([item[indexes]])))

        if time_index is not None:
            idx_x,idx_y = idx_list.index('x'),idx_list.index('y')
            forcing = self.get_forcings(time_index+self.step)
            for key,item in forcing.items():
                if key in features and key not in ['M_VT_x','M_VT_y']:
                    idx_list.append(f'{key}_f')
                    node_features.append(torch.tensor(item(node_features[idx_x],node_features[idx_y])))
                    #nan for some nodes on the mesh boundaries, quick fix -> replace with mean
                    if torch.isnan(node_features[-1]).any():
                        node_features[-1] = torch.nan_to_num(node_features[-1],torch.nanmean(node_features[-1]))
                        
                        

        node_features = torch.cat(node_features).t().to(torch.float32)
        return node_features, np.array(idx_list)
    
    

    def __get_element_edge_index(
            self,
            field: dict,
            indexes: np.array,
            edge_conectivity: bool = False
    ):
        """
        Function to get the edge features of a given field.

        Arguments:
            field: dict
                dictionary of data
            indexes: np.array
                indexes of the elements to sample from 
            edge_conectivity: bool
                if True, get the edge conectivity of the element

        returns:
            edge_index: torch.tensor
                tensor of edge features
        """
         #get t index of neighbours
        neighbours_t = field['t'][indexes]
        if edge_conectivity:
            edge_index_list = []
            for i,tri in enumerate(neighbours_t):
                #compute the vertex conectivity of the element with the rest of the elements
                vertex_conectivity = np.array([(tri[0]==neighbours_t).any(axis=1), (tri[1]==neighbours_t).any(axis=1), (tri[2]==neighbours_t).any(axis=1)]).sum(axis=0)
                #fetch elements sharing 2 vertexs - conected by and edge
                edge_elements = torch.tensor(np.where(vertex_conectivity==2)[0])
                #generate a tensor filled with element idx
                element_indeces= torch.full((len(edge_elements),),i)
                #stack the edge elements and the element index
                edge_index_list.append(
                    torch.stack([edge_elements,element_indeces], dim=0)
                ) 
            edge_index = torch.cat(edge_index_list,axis=1)

        else:
            #compute all adjacents edges #slow
            edge_list = []
            for i,element in enumerate(neighbours_t):
                adjacents = []
                for node in element:
                    adjacents += list(np.where(np.isin(neighbours_t,node))[0]) #adjacents per node
                adjacents_edges = [[i,j] for j in np.unique(adjacents) if j!=i] #pairs of all adj edges by element index
                edge_list += adjacents_edges
            edge_index = torch.tensor(edge_list).t()

        return edge_index
    
    def get_vertex_edge_index(
            self,
            vertex_neighbours: np.array,
            triangles: np.array
    ):
        """
        Function to get the edge features of a given field.

        Arguments:
            vertex_neighbours: np.array
                indexes of the vertexs to sample from
            triangles: np.array
                indexes of the triangles to sample from
        returns:
            edge_index: torch.tensor
                tensor of edge indeces
        """
        #Retrieve vertex conectivity from the mesh for the given vertexs 
        edge_index = np.concatenate([
            triangles.transpose()[:2],
            triangles.transpose()[1:],
            triangles.transpose()[[0,-1]]
        ],axis=-1)
        edge_index = np.unique(edge_index,axis=-1)   
       
        #select only the edges that are between the neighbour
        edge_index = [
            #np.where to get the index of the pair in neighbours (same index as node_features)
            [ np.where(vertex_neighbours==pair[0])[0], np.where(vertex_neighbours==pair[1])[0] ] 
            for pair in edge_index.transpose() 
            if np.isin(pair,vertex_neighbours).all(axis=-1)
        ]
     
        edge_index = torch.as_tensor(np.array(edge_index)).squeeze().t()

        return edge_index

    def __compute_edge_distances(
            self,
            feature_indeces: list[int],
            node_features: torch.tensor,
            edge_index: torch.tensor
    ):
        """
        Function to compute the distances between each edge.
        
        Arguments:
            feature_indeces: list[int]
                indexes of the features in the node features
            node_features: torch.tensor
                tensor of node features
            edge_features: torch.tensor
                tenso r of edge features
        returns:
            edge_dist: torch.tensor
                tensor of edge distances
        """



        coord_idx = [i for i,key in enumerate(feature_indeces) if key in ['x','y']]
        if len(coord_idx)==2:

            edges_coordinates = [
                torch.stack(
                    [
                        node_features[edge_row][:,coord_idx[0]],
                        node_features[edge_row][:,coord_idx[1]]
                    ]
                )
                for edge_row in edge_index
            ]
            edge_dist = torch.norm(edges_coordinates[1] - edges_coordinates[0],dim=0).unsqueeze(dim=-1).to(torch.float32) #convert to km
            #we also need to stack the node coordinates for the pos attribute
            positions = torch.stack(
                [
                    node_features[:,coord_idx[0]],
                    node_features[:,coord_idx[1]]
                ]
            )
        else:
            raise ValueError("Unable to find coordinates for nodes in mesh data. \nDid you include it in the feature list?")

        
        return edge_dist,positions
