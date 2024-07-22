
"""
Author: Francisco Amor
Date: May 2024

This script contains the GNN training pproceduredure.
The main function loads the trained model and the input data, train the model and save the best version of it.

Note: This code assumes that the necessary libraries and modules have been imported before running this script.

"""

import os
import copy
import random
import numpy as np
import pandas as pd
from tqdm import trange
import torch
import torch.optim as optim

from torch_geometric.loader import DataLoader

from models.MGN import MeshGraphNet,DiffusionModel,MeshGraphNetWithDiffusion
from models.GUnet import GUNet
from utils.metrics import StepForwardLoss,Constrained_loss,Vector_loss, Deformation_loss
from utils.graph_utils import compute_stats_batch
from utils.graph_utils import normalize_data
import wandb
import yaml




def train(dataset, model, device, stats_list, args):
    df = pd.DataFrame(columns=['epoch', 'train_loss', 'test_loss'])
    model_name = f"{args.dataset_dir.split('/')[-1]}_{args.model_type}_{args.loss}_{args.loss_args}_{args.opt_scheduler}_nl{args.num_layers}_bs{args.batch_size}_hd{args.hidden_dim}_ep{args.epochs}_wd{args.weight_decay}_lr{args.lr}_shuff_{args.shuffle}_tr{args.train_size}_te{args.val_size}_acc{args.acc}"
    print(model_name)

    loader = DataLoader(dataset[:args.train_size], batch_size=args.batch_size, shuffle=args.shuffle)
    val_loader = DataLoader(dataset[args.train_size:], batch_size=args.batch_size, shuffle=False)
    mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y = [i.to(device) for i in stats_list]
    model = model.to(device)
    scheduler, opt = build_optimizer(args, model.parameters())
    loss_fn = build_loss(args)
    
    losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model = None
    
    for epoch in trange(args.epochs, desc="Training", unit="epoch"):
        model.train()
        total_loss = train_epoch(loader, model, loss_fn, opt, device, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y,args.acc)
        losses.append(total_loss)
        
        val_loss = validate_epoch(val_loader, model, loss_fn, device, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y,args.acc)
        val_losses.append(val_loss)
        
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                # For ReduceLROnPlateau, pass the validation loss as an argument
                scheduler.step(val_loss)
            else:
                # For other types of schedulers, call step without arguments
                scheduler.step()
        
        if args.wandb_log:
            wandb.log({'val_loss': val_loss, 'train_loss': total_loss})
        
        best_val_loss = update_checkpoints(epoch, args, model_name, best_val_loss, val_loss, model, df, losses, val_losses,stats_list)
        
        if (epoch % 2 == 0):
            print(f"Epoch {epoch}: Train Loss {total_loss:.5f}, Validation Loss {val_loss:.5f}")
    
    return val_losses, losses, best_model, best_val_loss, val_loader, model_name

def update_checkpoints(epoch, args, model_name, best_val_loss, val_loss, model, df, losses, val_losses,stats_list):
    df = pd.concat([df, pd.DataFrame({'epoch': [epoch], 'train_loss': [losses[-1]], 'test_loss': [val_losses[-1]]})])
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)
        if args.save_best_model and best_model:
            save_path = os.path.join(args.checkpoint_dir, model_name + '.pt')
            torch.save(best_model.state_dict(), save_path)
            df.to_csv(os.path.join(args.checkpoint_dir, model_name + '.csv'), index=False)
    

    return best_val_loss

def train_epoch(loader, model, loss_fn, optimizer, device, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y,acc):
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        
        #check if batch.mask is defined
        if hasattr(batch, 'mask'):
            mask = batch.mask == 0
        else:
            mask = batch.x[:, -1] == 0
        if acc:
            batch.y = batch.x[:,:2] - batch.y
        batch = normalize_data(batch, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y)
        optimizer.zero_grad()
        pred = model(batch)
        if isinstance(loss_fn, StepForwardLoss):
            loss = loss_fn(pred, batch.y, batch.x[:,:2],mask)
        elif isinstance(loss_fn, Vector_loss) or isinstance(loss_fn, Deformation_loss):
            loss = loss_fn(pred[mask], batch.y[mask], batch.mesh_pos[mask])
        else:
            loss = loss_fn(pred[mask], batch.y[mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate_epoch(val_loader, model, loss_fn, device, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y,acc):
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            #node_type mask
            if hasattr(batch, 'mask'):
                mask = batch.mask == 0
            else:
                mask = batch.x[:, -1] == 0
            if acc:
                batch.y = batch.x[:,:2] - batch.y
            batch = normalize_data(batch, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y)
            pred = model(batch)
            if isinstance(loss_fn, StepForwardLoss):
                val_loss += loss_fn(pred, batch.y, batch.x[:,:2],mask).item()
            elif isinstance(loss_fn, Vector_loss) or isinstance(loss_fn, Deformation_loss):
                val_loss += loss_fn(pred[mask], batch.y[mask], batch.mesh_pos[mask]).item()
            else:
                val_loss += loss_fn(pred[mask], batch.y[mask]).item()

    return val_loss / len(val_loader)



def build_optimizer(args, params):
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p : p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adamW':
        optimizer = optim.AdamW(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.opt_restart)
    elif args.opt_scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    elif args.opt_scheduler == 'exp':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    return scheduler, optimizer


def build_loss(args):
    if args.loss == 'mse':
        return torch.nn.MSELoss()
    elif args.loss == 'l1':
        return torch.nn.L1Loss()
    elif args.loss == 'huber':
        return torch.nn.HuberLoss()
    elif args.loss == 'stepForwardMSE':
        return StepForwardLoss()
    elif args.loss == 'stepForwardL1':
        return StepForwardLoss(loss=torch.nn.L1Loss())
    elif args.loss == 'constrainet_loss_v2':
        return Constrained_loss(alpha=args.loss_args)
    elif args.loss == 'vector_loss':
        return Vector_loss(args.loss_args)
    elif args.loss == 'deformation_loss':
        return Deformation_loss(loss=torch.nn.MSELoss(),alpha=args.loss_args)

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d



def main(
    dataset_dir='../data_graphs/crop_vel_3h_survived_f',
    model_type='meshgraphnet',
    num_layers=8,
    batch_size=4,
    hidden_dim=16,
    epochs=60,
    opt='adamW',
    opt_scheduler='plateau',
    opt_restart=0,
    loss='mse',
    acc = False,
    loss_args = 1e-15,
    weight_decay=1e-5,
    lr=0.00062,
    train_size=250,
    val_size=48,
    device='cuda',
    shuffle=True,
    save_best_model=True,#!!!!!!!!!!!!!!!
    checkpoint_dir='../comparissons/',
    postprocess_dir='./2d_loss_plots/',
    wandb_sweep: bool = True,
    wandb_log: bool = True
):

    #wandb init
    if wandb_log:
        wandb.login()

        if wandb_sweep:
            with open('./sweep_config.yml', 'r') as file:
                wd_config = yaml.safe_load(file)
            
            run = wandb.init(config=wd_config, entity='franamor98')

            lr = wandb.config.lr
            model_type = wandb.config.model_type
            weight_decay = wandb.config.weight_decay
            dataset_dir = wandb.config.dataset_dir
            num_layers = wandb.config.num_layers
            batch_size = wandb.config.batch_size
            hidden_dim = wandb.config.hidden_dim
            loss = wandb.config.loss
            loss_args = wandb.config.loss_args

        else:
            run = wandb.init(project='nextsim', entity='franamor98')

    for args in [
            {   
                'dataset_dir': dataset_dir,
                'model_type': model_type,
                'num_layers': num_layers,
                'batch_size': batch_size,
                'hidden_dim': hidden_dim,
                'epochs': epochs,
                'opt': opt,
                'opt_scheduler': opt_scheduler,
                'opt_restart': opt_restart,
                'loss': loss,
                'loss_args': loss_args,
                'weight_decay': weight_decay,
                'lr': lr,
                'train_size': train_size,
                'val_size': val_size,
                'device': device,
                'shuffle': shuffle,
                'save_best_model': save_best_model,
                'checkpoint_dir': checkpoint_dir,
                'postprocess_dir': postprocess_dir,
                "opt_decay_step": 10,
                "wandb_log": wandb_log,
                "acc": acc
            },
        ]:
            args = objectview(args)
    print(args.lr,args.weight_decay)
    

    torch.manual_seed(5)
    random.seed(5)
    np.random.seed(5)

    # Load dataset
    graph_files = [i for i in os.listdir(dataset_dir) if "list" not in i and "pt" in i]
    graph_files = sorted(graph_files,key=lambda x:int(x.split("_")[-1].split(".")[0]))
    print("Number of files in dataset: ", len(graph_files))
    try:
        graph_files = graph_files[:train_size + val_size]
        
    except:
        print('Dataset is smaller than train_size + val_size')

    dataset = []
    for file in graph_files:
        with open(os.path.join(dataset_dir,file),'rb') as f:
            try:
                dataset.append(torch.load(f))
            except:
                print(f"Error loading file {file}")
  
    print(len(dataset))

    # Create checkpoint directory if it does not exist
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    
    # Compute stats
    stats_list = compute_stats_batch(dataset,acc)

    if args.shuffle:
        pass#random.shuffle(dataset)

    #create the model
    num_node_features = dataset[0].x.shape[1]
    num_edge_features = dataset[0].edge_attr.shape[1]
    num_classes = dataset[0].y.shape[1] # the dynamic variables have the shape of 2 (velocity)

    if args.model_type == 'meshgraphnet': #MGN model
        model = MeshGraphNet(num_node_features, num_edge_features, args.hidden_dim, num_classes,
                            args)
    elif args.model_type == 'gunet': #GUNet model
        model = GUNet(num_node_features,hidden_dimension=args.hidden_dim, num_classes= num_classes,depth=args.num_layers)

    elif args.model_type == 'diff_mgn': #experimental model using diffusion concepts... not fully developed
        diffusion_block = DiffusionModel(num_node_features,num_node_features,num_node_features)
        mgn = MeshGraphNet(num_node_features, num_edge_features, args.hidden_dim, num_classes,args)
        model = MeshGraphNetWithDiffusion(mgn, diffusion_block,.1,1)

    # Train
    val_losses, losses, best_model, best_val_loss, test_loader,model_name = train(dataset,model, device, stats_list,args)

    # Log metrics to WandB 
    
    if args.wandb_log:
        wandb.log({
            'min_test_loss': min(val_losses),
            'min_loss': min(losses),
            'best_model': model_name,
        })
    

    print("Min test set loss: {0}".format(min(val_losses)))
    print("Minimum loss: {0}".format(min(losses)))
    print("Best model: {0}".format(model_name))
    


if __name__ == "__main__":

    main()


