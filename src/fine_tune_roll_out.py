"""
Author: Francisco Amor
Date: May 2024

This script contains the implementation of the roll_out training procedure.
The main function loads the trained model and the input data, and then calls the roll_day function to perform the simulation.

Note: This code assumes that the necessary libraries and modules have been imported before running this script.

"""

import torch
import random
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch.optim as optim
import copy


import os
import numpy as np
from torch_geometric.data import Data
from scipy.spatial import Delaunay
#NEAREST NEUGHBOURS interpolatior
from scipy.interpolate import NearestNDInterpolator


import sys
sys.path.append('../src')
from ice_graph.ice_graph import Ice_graph
from utils.graph_utils import compute_stats_batch,normalize_data,normalize,unnormalize
from models.MGN import MeshGraphNet
from models.GUnet import GUNet




def triangles_to_edges(faces):
    """Computes mesh edges from triangles."""
    # collect edges from triangles
    faces = torch.tensor(faces)
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
    return torch.stack([torch.cat((senders, receivers), dim=0),
            torch.cat((receivers, senders), dim=0)])


def get_interpolators(data):
    x,y = data.mesh_pos[:,0],data.mesh_pos[:,1]
    features = data.x.cpu().numpy()
    interp = {}
    interp['u_wind'] = NearestNDInterpolator((x, y), features[:,2])
    interp['v_wind'] = NearestNDInterpolator((x, y), features[:,3])
    interp['u_ocean'] = NearestNDInterpolator((x, y), features[:,4])
    interp['v_ocean'] = NearestNDInterpolator((x, y), features[:,5])
    interp['u_wind1'] = NearestNDInterpolator((x, y), features[:,8])
    interp['v_wind1'] = NearestNDInterpolator((x, y), features[:,9])
    interp['u_ocean1'] = NearestNDInterpolator((x, y), features[:,10])
    interp['v_ocean1'] = NearestNDInterpolator((x, y), features[:,11])

    return interp


def jacobian(x0, y0, x1, y1, x2, y2):
    return (x1-x0)*(y2-y0)-(x2-x0)*(y1-y0)


def compute_div(p0,p1, lower_threshold, upper_threshold):
    x0,y0 = p0[:,0].cpu().numpy(),p0[:,1].cpu().numpy()
    x1,y1 = p1[:,0].cpu().numpy(),p1[:,1].cpu().numpy()
    pos = np.stack([x1,y1],axis=1)
    t1 = Delaunay(pos,qhull_options='QJ').simplices
    # find starting / ending coordinates for each elements
    x0a, x0b, x0c = x0[t1].T
    y0a, y0b, y0c = y0[t1].T
    xpa, xpb, xpc = x1[t1].T
    ypa, ypb, ypc = y1[t1].T

    # compute area at the first and second snapshots (subsampled mesh)
    a0 = jacobian(x0a, y0a, x0b, y0b, x0c, y0c)
    ap = jacobian(xpa, ypa, xpb, ypb, xpc, ypc)

    #divergence
    div_t = a0/ap

    #interpolate div into x,y grid
    x_mean = np.mean(x1[t1],axis=1)
    y_mean = np.mean(y1[t1],axis=1)
    div_interp = NearestNDInterpolator((x_mean, y_mean), div_t)
    div = div_interp(x1,y1)

    filtered_div = np.clip(np.where((div < lower_threshold) | (div > upper_threshold), div, 1),0.9,1.1)


    return filtered_div, t1


class objectview(object):
    def __init__(self, d):
        self.__dict__ = d




def roll_day(data_list, model, device,stats_list,dt=3600*3):
    
    data_1 = None
    true_pos = None
    mesh_pos = None
    true_pos = None
    mask = None
    mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge,mean_vec_y,std_vec_y = stats_list
    for i,data in enumerate(data_list[:-1]):

        if data_1 is not None:
            data = data_1

        data = data.to(device)
        if data_1 is None:
            mask = data.x[:, -1] == 0

        data = normalize_data(data,mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge,mean_vec_y,std_vec_y)
    
        pred = model(data)
        pred = unnormalize(pred,mean_vec_y,std_vec_y)
        target = unnormalize(data.y,mean_vec_y,std_vec_y)
        data.x = unnormalize(data.x,mean_vec_x,std_vec_x)

        #inputs --> [dx01, dy01, wind_u, wind_v, ocean_u, ocean_v, concentration, thickness, wind_u1,wind_v1,ocean_u1,ocean_v1, node_type]
        #compute future positions
        mesh_pos = data.mesh_pos + pred * dt

        if true_pos is not None:
            true_pos = true_pos + target * dt
        else:
            true_pos = data.mesh_pos + target * dt

        
        features = torch.zeros_like(data.x)
        positions = mesh_pos.detach().cpu()
        #update ocean and wind data.x[:,2:6] + data.x[:,8:12]
        interp = get_interpolators(copy.deepcopy(data_list[i+1]))
        features[:,2] = torch.tensor(interp['u_wind'](positions[:,0],positions[:,1]))
        features[:,3] = torch.tensor(interp['v_wind'](positions[:,0],positions[:,1]))
        features[:,4] = torch.tensor(interp['u_ocean'](positions[:,0],positions[:,1]))
        features[:,5] = torch.tensor(interp['v_ocean'](positions[:,0],positions[:,1]))
        features[:,8] = torch.tensor(interp['u_wind1'](positions[:,0],positions[:,1]))
        features[:,9] = torch.tensor(interp['v_wind1'](positions[:,0],positions[:,1]))
        features[:,10] = torch.tensor(interp['u_ocean1'](positions[:,0],positions[:,1]))
        features[:,11] = torch.tensor(interp['v_ocean1'](positions[:,0],positions[:,1]))
        
        #update vel
        features[:,:2] = pred

        #get divergence div, t0 update
        positions0 = data.mesh_pos.detach()
        div, t1 = compute_div(positions0,positions,0.997 , 1.003)

        div = torch.tensor(div).type(torch.float).to(device)

        features[:,6] = torch.clip(data.x[:,6]*div,0,1)
        features[:,7] = data.x[:,7]*div
        features[:,-1] = features[:,-1]
        
        
        #triangulate t0
        edge_index = triangles_to_edges(t1).type(torch.long)

        u_i = torch.stack(
            (data.mesh_pos[:,0][edge_index[0]],
            data.mesh_pos[:,1][edge_index[0]]),
            dim=1
        )
        u_j = torch.stack(
            (data.mesh_pos[:,0][edge_index[1]],
            data.mesh_pos[:,1][edge_index[1]]),
            dim=1
        )

        u_ij = u_i - u_j
        u_ij_norm = torch.norm(u_ij, p=2, dim=1, keepdim=True)
        edge_attr = torch.cat((u_ij, u_ij_norm), dim=-1).type(torch.float)

        #interpolate next y values into out mesh
        data_true = data_list[i+1]
        interp = NearestNDInterpolator((data_true.mesh_pos[:,0],data_true.mesh_pos[:,1]),data_true.y.cpu().numpy())
        y = torch.tensor(interp(positions[:,0],positions[:,1])).type(torch.float)
            
        
        data_1 = Data(x=features, edge_index=edge_index,edge_attr=edge_attr, mesh_pos = mesh_pos, y =y)

    return mesh_pos, true_pos, mask
    

def main():
    

    for args in [
            {'model_type': 'meshgraphnet',
            'num_layers': 8,
            'batch_size': 8,
            'hidden_dim': 10,
            'epochs': 1,
            'opt': 'adam',
            'opt_scheduler': 'cos',
            'opt_restart': 0,
            'weight_decay': 5e-4,
            'lr': 0.001,
            'train_size': 40,
            'test_size': 2,
            'device':'cpu',
            'shuffle': True,
            'save_velo_val': False,
            'save_best_model': True,
            'checkpoint_dir': './best_models/',
            'postprocess_dir': './2d_loss_plots/'},
        ]:
            args = objectview(args)

    #To ensure reproducibility the best we can, here we control the sources of
    #randomness by seeding the various random number generators used in this Colab
    #For more information, see: https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(5)  #Torch
    random.seed(5)        #Python
    np.random.seed(5)     #NumPy


    # load model.
    #load graphs one by one
    file_path = '../data_graphs/vel_3h_survived_f'
    #get file names sorted in order
    graph_files = [i for i in os.listdir(file_path) if "list" not in i and "pt" in i]
    graph_files = sorted(graph_files,key=lambda x:int(x.split("_")[-1].split(".")[0]) if x.split("_")[-1].split(".")[0].isdigit() else 0)[:]
    graph_list = []
    for file in tqdm(graph_files,desc="Loading graphs"):
        with open(os.path.join(file_path,file),'rb') as f:
                graph_list.append(torch.load(f))
                
    graph_list[0], len(graph_list)

    stats_list = compute_stats_batch(graph_list)
    dataset = graph_list

    model_name = "vel_3h_survived_f_meshgraphnet_mse_0_plateau_nl8_bs8_hd10_ep120_wd1e-05_lr0.00062_shuff_True_tr250_te20"
    args.device = torch.device('cuda') # 
    num_node_features = dataset[0].x.shape[1]
    num_edge_features = dataset[0].edge_attr.shape[1]
    num_classes = 2 # the dynamic variables have the shape of 2 (velocity)
    PATH = f"../best_models2/{model_name}.pt"#os.path.join( checkpoint_dir, f'{model_name}.pt')

    if args.model_type == 'meshgraphnet':
            model = MeshGraphNet(num_node_features, num_edge_features, args.hidden_dim, num_classes,
                                args)
    if args.model_type == 'gunet':
        model = GUNet(num_node_features, num_classes)

    model.load_state_dict(torch.load(PATH, map_location=args.device))

    #both components
    lead = 3
    dt = 3600*lead
    snapshot_per_day = int(24/lead)

    [mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge,mean_vec_y,std_vec_y] = stats_list
    loss_fn = nn.MSELoss()
    device = torch.device('cuda')
    stats_list = [mean_vec_x.to(device),std_vec_x.to(device),mean_vec_edge.to(device),std_vec_edge.to(device),mean_vec_y.to(device),std_vec_y.to(device)] 
    model = model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    n_epochs = 100


    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.decoder.parameters():
        param.requires_grad = True

    data_per_step = copy.deepcopy(dataset[::int(dt/1800)])
    #last day as val
    val_roll_out_set = data_per_step[-snapshot_per_day:]
    train_roll_out_set = data_per_step[:-snapshot_per_day]
    #split rollout into sets of size 8
    train_roll_out = [train_roll_out_set[i:i+snapshot_per_day] for i in range(0,len(train_roll_out_set),snapshot_per_day)]
    train_losses = []
    val_losses = []
    for epochs in range(n_epochs):

       
        roll_out_set = copy.deepcopy(dataset[::int(dt/1800)])

        print(f"Epoch {epochs}")
        
        print("\tTraining")
        total_loss = 0
        for i,roll_out_set in tqdm(enumerate(train_roll_out)):
            optimizer.zero_grad()
            mesh_pos, true_pos, mask = roll_day(copy.deepcopy(roll_out_set), model, device,stats_list,dt=dt)
            loss = loss_fn(mesh_pos[mask], true_pos[mask])
            loss.backward()
            optimizer.step()
            total_loss += loss

        total_loss = total_loss/len(train_roll_out)
        train_losses.append(total_loss)
        print(f" \tTrain Loss: {torch.sqrt(total_loss)}")
        
        print("\tValidation")
        optimizer.zero_grad()

        mesh_pos, true_pos, mask = roll_day(copy.deepcopy(val_roll_out_set), model, device,stats_list,dt=dt)
        loss = loss_fn(mesh_pos[mask], true_pos[mask])
        scheduler.step()
        print(f"\tVal Loss: {torch.sqrt(loss)}")
        val_losses.append(loss)



        #save model if best
        if len(val_losses) > 1:
            if val_losses[-1] < min(val_losses[:-1]):
                print("Saving model")
                torch.save(model.state_dict(),f"../roll_out_models/{model_name}_rollout2.pt")

if __name__ == '__main__':
    main()


