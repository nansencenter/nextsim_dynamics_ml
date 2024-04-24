import os
import numpy as np
import torch
from tqdm import trange
import sys
from torch_geometric.data import Data

sys.path.append('../src')
from ice_graph.ice_graph import Ice_graph
import multiprocessing as mp



def triangles_to_edges(faces):
    """Computes mesh edges from triangles."""
    # collect edges from triangles
    faces = torch.tensor(np.array(faces))
    edges = torch.cat((faces[:, 0:2],
                       faces[:, 1:3],
                       torch.stack((faces[:, 2], faces[:, 0]), dim=1)), dim=0)
    
    # those edges are sometimes duplicated (within the mesh) and sometimes
    # single (at the mesh boundary).
    # sort & pack edges as single torch.int64
    receivers = torch.min(edges, dim=1)[0]
    senders = torch.max(edges, dim=1)[0]
    packed_edges = torch.stack((senders, receivers), dim=1).to(torch.int64)
    # remove duplicates and unpack
    unique_edges = torch.unique(packed_edges, dim=0)
    senders, receivers = unique_edges.unbind(1)
    
    # create two-way connectivity
    return torch.stack((torch.cat((senders, receivers), dim=0),
            torch.cat((receivers, senders), dim=0)))



def generate_full_graph(time, nextsim, delta_t,out_path,start_time):
    forcing_interp = nextsim.get_forcings(time-1,['M_wind_x', 'M_wind_y', 'M_ocean_x', 'M_ocean_y', 'M_VT_x', 'M_VT_y','Concentration','Thickness'])
    vertex_data = nextsim.get_item(time, elements=False)
    element_data = nextsim.get_item(time, elements=True)

    wind_u = torch.tensor(forcing_interp['M_wind_x'](vertex_data['x'],vertex_data['y']))
    wind_v = torch.tensor(forcing_interp['M_wind_y'](vertex_data['x'],vertex_data['y']))
    ocean_u = torch.tensor(forcing_interp['M_ocean_x'](vertex_data['x'],vertex_data['y']))
    ocean_v = torch.tensor(forcing_interp['M_ocean_y'](vertex_data['x'],vertex_data['y']))
    ice_u = torch.tensor(forcing_interp['M_VT_x'](vertex_data['x'],vertex_data['y']))
    ice_v = torch.tensor(forcing_interp['M_VT_y'](vertex_data['x'],vertex_data['y']))
    concentration = torch.tensor(forcing_interp['Concentration'](vertex_data['x'],vertex_data['y']))
    thickness = torch.tensor(forcing_interp['Thickness'](vertex_data['x'],vertex_data['y']))

    # Replace nan with 0
    wind_u = torch.nan_to_num(wind_u)
    wind_v = torch.nan_to_num(wind_v)
    ocean_u = torch.nan_to_num(ocean_u)
    ocean_v = torch.nan_to_num(ocean_v)
    ice_u = torch.nan_to_num(ice_u)
    ice_v = torch.nan_to_num(ice_v)
    concentration = torch.nan_to_num(concentration)
    thickness = torch.nan_to_num(thickness)

    x = torch.stack((ice_u, ice_v, wind_u, wind_v, ocean_u, ocean_v, concentration, thickness), dim=1).type(torch.float)

    edge_index = triangles_to_edges(element_data['t']).type(torch.long)

    u_i = torch.stack(
        (torch.tensor(vertex_data['x'])[edge_index[0]],
         torch.tensor(vertex_data['y'])[edge_index[0]]),
        dim=1
    )
    u_j = torch.stack(
        (torch.tensor(vertex_data['x'])[edge_index[1]],
         torch.tensor(vertex_data['y'])[edge_index[1]]),
        dim=1
    )
    u_ij = u_i - u_j
    u_ij_norm = torch.norm(u_ij, p=2, dim=1, keepdim=True)
    edge_attr = torch.cat((u_ij, u_ij_norm), dim=-1).type(torch.float)

    v_t1 = torch.stack(
        (torch.tensor(vertex_data['M_VT_x']),
         torch.tensor(vertex_data['M_VT_y'])),
        dim=1
    )
    v_t0 = torch.stack((torch.tensor(ice_u), torch.tensor(ice_v)), dim=1)
    y = ((v_t1 - v_t0) / delta_t).type(torch.float)

    # Data needed for visualization code
    cells = torch.tensor(element_data['t']).type(torch.float)
    mesh_pos = torch.stack((torch.tensor(vertex_data['x']), torch.tensor(vertex_data['y'])), dim=1).type(torch.float)

    data =  Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, cells=cells, mesh_pos=mesh_pos)
    file_name = f"graph_{time+start_time}.pt"
    #save in path
    torch.save(data, os.path.join(out_path, file_name))

    del data

    return file_name


def generate_centered_graph(time, nextsim, vertex_i,delta_t,out_path,start_time):


    forcing_interp = nextsim.get_forcings(time-1,['M_wind_x', 'M_wind_y', 'M_ocean_x', 'M_ocean_y', 'M_VT_x', 'M_VT_y','Concentration','Thickness'])
    vertex_data = nextsim.get_item(time,elements=False)
    element_data = nextsim.get_item(time,elements=True)
    vertex_index = np.where(np.isin(vertex_data['i'],vertex_i))[0]
    elements_index = np.where(np.isin(element_data['t'],vertex_index))[0]

    wind_u = torch.tensor(forcing_interp['M_wind_x'](vertex_data['x'][vertex_index],vertex_data['y'][vertex_index]))
    wind_v = torch.tensor(forcing_interp['M_wind_y'](vertex_data['x'][vertex_index],vertex_data['y'][vertex_index]))
    ocean_u = torch.tensor(forcing_interp['M_ocean_x'](vertex_data['x'][vertex_index],vertex_data['y'][vertex_index]))
    ocean_v = torch.tensor(forcing_interp['M_ocean_y'](vertex_data['x'][vertex_index],vertex_data['y'][vertex_index]))
    ice_u = torch.tensor(vertex_data['M_VT_x'][vertex_index])
    ice_v = torch.tensor(vertex_data['M_VT_y'][vertex_index])
    concentration = torch.tensor(forcing_interp['Concentration'](vertex_data['x'][vertex_index],vertex_data['y'][vertex_index]))
    thickness = torch.tensor(forcing_interp['Thickness'](vertex_data['x'][vertex_index],vertex_data['y'][vertex_index]))

    #replace nan to 0, bad practice
    wind_u = torch.nan_to_num(wind_u)
    wind_v = torch.nan_to_num(wind_v)
    ocean_u = torch.nan_to_num(ocean_u)
    ocean_v = torch.nan_to_num(ocean_v)
    ice_u = torch.nan_to_num(ice_u)
    ice_v = torch.nan_to_num(ice_v)
    concentration = torch.nan_to_num(concentration)
    thickness = torch.nan_to_num(thickness)


    x = torch.stack((ice_u,ice_v,wind_u,wind_v,ocean_u,ocean_v,concentration,thickness),dim=1).type(torch.float)
    
    edge_index = triangles_to_edges(element_data['t'][elements_index]).type(torch.long)

    u_i = torch.stack(
        (torch.tensor(vertex_data['x'])[edge_index[0]],
        torch.tensor(vertex_data['y'])[edge_index[0]]),
        dim=1
    )
    u_j = torch.stack(
        (torch.tensor(vertex_data['x'])[edge_index[1]],
        torch.tensor(vertex_data['y'])[edge_index[1]]),
        dim=1
    )
    u_ij=u_i-u_j
    u_ij_norm = torch.norm(u_ij,p=2,dim=1,keepdim=True)
    edge_attr = torch.cat((u_ij,u_ij_norm),dim=-1).type(torch.float)

    v_t1 = torch.stack(
        (torch.tensor(vertex_data['M_VT_x'][vertex_index]),
        torch.tensor(vertex_data['M_VT_y'][vertex_index])),
        dim=1
    )
    v_t0=torch.stack((torch.tensor(ice_u),torch.tensor(ice_v)),dim=1)
    y = ((v_t1 - v_t0) / delta_t).type(torch.float)

    
    #Data needed for visualization code
    cells = torch.tensor(element_data['t'][vertex_index]).type(torch.float)
    mesh_pos = torch.stack((torch.tensor(vertex_data['x'][vertex_index]),torch.tensor(vertex_data['y'][vertex_index])),dim=1).type(torch.float)

    data =  Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, cells=cells, mesh_pos=mesh_pos)

    file_name = f"graph_{time+start_time}.pt"
    #save in path
    torch.save(data, os.path.join(out_path, file_name))

    del data
    del element_data
    del vertex_data

    return file_name



def main(
        data_path: str = '../../week_data',
        out_path: str = '../data_graphs/centered_graphs',
        save_name: str = 'graph_list.pt',
        delta_t: int = 1800,
        start_time: int = 48,
        end_time: int = 100,
        center: bool = True,
        neigh_layers: int = 20
):

    # Create output directory if it does not exist
    os.makedirs(out_path, exist_ok=True)

    # Load files
    try:
        files = sorted(os.listdir(data_path))[start_time:end_time]
    except:
        print(f"Can't open directory: {data_path} or time range is invalid.")
        return

    print(f'{len(files)} existing files')
    file_graphs = []
    for file in files:
        with open(f"{data_path}/{file}", 'rb') as f:
            try:
                file_graphs.append(dict(np.load(f)))
            except:
                print(f"Can't open file: {file}")


    print(f'Loaded {len(file_graphs)} files')

    nextsim = Ice_graph(
        file_graphs,
        vertex_element_features=['M_wind_x', 'M_wind_y', 'M_ocean_x', 'M_ocean_y', 'M_VT_x', 'M_VT_y', 'x', 'y']
    )


    graph_list = []
    if center:

        files = sorted(os.listdir(data_path))[48]
        with open(f"{data_path}/{files}", 'rb') as f:
            try:
                file_graph =dict(np.load(f))
            except:
                print(f"Can't open file: {file}")
        nextsim_load_index = Ice_graph(
            [file_graph],
            vertex_element_features=['M_wind_x', 'M_wind_y', 'M_ocean_x', 'M_ocean_y', 'M_VT_x', 'M_VT_y', 'x', 'y']
        )

        _,_,samples  = nextsim_load_index.compute_vertex_neighbourhood(vertex_index=8544,time_index=0,return_vertex=True,n_neighbours=neigh_layers)
        selected_graph = nextsim_load_index.get_item(0,elements=False) 
        vertex_i = selected_graph['i'][samples]
        for time in trange(1, len(file_graphs)):
            graph_list.append(generate_centered_graph(time, nextsim, vertex_i,delta_t,out_path,start_time))
    else:
        for time in trange(1, len(file_graphs)):
            graph_list.append(generate_full_graph(time, nextsim, delta_t,out_path,start_time))

    torch.save(graph_list, os.path.join(out_path, save_name))


if __name__ == '__main__':
    main()