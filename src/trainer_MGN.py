import os
import copy
import random
import numpy as np
import pandas as pd
from tqdm import trange
import torch
import torch.optim as optim

from torch_geometric.loader import DataLoader

from models.MGN import MeshGraphNet
from models.GUnet import GUNet
from utils.graph_utils import compute_stats_batch
from utils.graph_utils import normalize_data
import wandb
import yaml

def train(dataset, model, device, stats_list, args):
    '''
    Performs a training loop on the dataset for MeshGraphNets. Also calls
    test and validation functions.
    '''

    df = pd.DataFrame(columns=['epoch','train_loss','test_loss', 'velo_val_loss'])

    #Define the model name for saving
    model_name='model_nl'+str(args.num_layers)+'_bs'+str(args.batch_size) + \
               '_hd'+str(args.hidden_dim)+'_ep'+str(args.epochs)+'_wd'+str(args.weight_decay) + \
               '_lr'+str(args.lr)+'_shuff_'+str(args.shuffle)+'_tr'+str(args.train_size)+'_te'+str(args.val_size)

    #torch_geometric DataLoaders are used for handling the data of lists of graphs
    loader = DataLoader(dataset[:args.train_size], batch_size=args.batch_size, shuffle=args.shuffle)
    val_loader = DataLoader(dataset[args.train_size:], batch_size=args.batch_size, shuffle=args.shuffle)

    #The statistics of the data are decomposed and moved to device
    [mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge,mean_vec_y,std_vec_y] = stats_list
    (mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge,mean_vec_y,std_vec_y)=(mean_vec_x.to(device),
        std_vec_x.to(device),mean_vec_edge.to(device),std_vec_edge.to(device),mean_vec_y.to(device),std_vec_y.to(device))

    model = model.to(device)
   
    scheduler, opt = build_optimizer(args, model.parameters())
    loss_fn = build_loss(args)

    # train
    losses = []
    val_losses = []
    best_val_loss = np.inf
    best_model = None
    for epoch in trange(args.epochs, desc="Training", unit="Epochs"):
        total_loss = 0
        model.train()
        num_loops=0
        for batch in loader:
            #to(device) moves the data to the device (CPU or GPU)
            batch=batch.to(device)
            #normalize data
            batch = normalize_data(batch,mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge,mean_vec_y,std_vec_y)
            opt.zero_grad()         #zero gradients each time
            pred = model(batch)
            loss = loss_fn(pred, batch.y)
            loss.backward()         #backpropagate loss
            opt.step()
            total_loss += loss.item()
            num_loops+=1
        total_loss /= num_loops
        losses.append(total_loss)


        #Every x epoch, calculate val loss, save metrics and get best model
        if epoch % 1 == 0:

            #test_loss = test(test_loader,device,model,mean_vec_x,std_vec_x,mean_vec_edge,
            #                    std_vec_edge,mean_vec_y,std_vec_y)
            val_loss  = 0
            num_loops=0
            for batch in val_loader:
                batch = batch.to(device)
                batch = normalize_data(batch,mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge,mean_vec_y,std_vec_y)
                pred = model(batch)
                with torch.no_grad():
                    val_loss += loss_fn(pred, batch.y)
                num_loops+=1
            val_loss /= num_loops

            val_losses.append(val_loss.item())
            scheduler.step(val_loss.item()) if scheduler else None

            if args.wandb_log:
                wandb.log({
                    'test_loss': val_loss.item(),
                    'train_loss': total_loss
                })
                
            # saving model
            if not os.path.isdir(args.checkpoint_dir ):
                os.mkdir(args.checkpoint_dir)

            PATH = os.path.join(args.checkpoint_dir, model_name+'.csv')
            df.to_csv(PATH,index=False)

            #save the model if the current one is better than the previous best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(model)

        else:
            #If not the tenth epoch, append the previously calculated loss to the
            #list in order to be able to plot it on the same plot as the training losses
            val_losses.append(val_losses[-1])

        df = pd.concat([df, pd.DataFrame({'epoch': [epoch], 'train_loss': losses[-1:], 'test_loss': val_losses[-1:]})] )
        
        if(epoch%2==0):
            
            print("train loss", str(round(total_loss,2)), "test loss", str(round(val_loss.item(),2)))

            if(args.save_best_model):

                PATH = os.path.join(args.checkpoint_dir, model_name+'.pt')
                torch.save(best_model.state_dict(), PATH )

    return val_losses, losses, best_model, best_val_loss, val_loader,model_name

def test(loader,device,test_model,
         mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge,mean_vec_y,std_vec_y):

    '''
    Calculates test set losses .
    '''

    loss=0
    num_loops=0

    for data in loader:
        data=data.to(device)
        with torch.no_grad():
            #calculate the loss for the model given the test set
            pred = test_model(data)
            loss += test_model.loss(pred, data,mean_vec_y,std_vec_y)
        num_loops+=1

    return loss/num_loops

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
    return scheduler, optimizer


def build_loss(args):
    if args.loss == 'mse':
        return torch.nn.MSELoss()
    elif args.loss == 'l1':
        return torch.nn.L1Loss()
    elif args.loss == 'huber':
        return torch.nn.HuberLoss()

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d



def main(
    dataset_dir='../data_graphs/centered_graphs',
    model_type='gunet',
    num_layers=10,
    batch_size=4,
    hidden_dim=10,
    epochs=10,
    opt='adamW',
    opt_scheduler='plateau',
    opt_restart=0,
    loss='mse',
    weight_decay=5e-4,
    lr=0.0001,
    train_size=260,
    val_size=60,
    device='cpu',
    shuffle=True,
    save_best_model=True,
    checkpoint_dir='../best_models/',
    postprocess_dir='./2d_loss_plots/',
    wandb_sweep: bool = False,
    wandb_log: bool = False
):

    #wandb init
    if wandb_log:
        wandb.login()

        if wandb_sweep:
            with open('./sweep_config.yml', 'r') as file:
                wd_config = yaml.safe_load(file)
            
            run = wandb.init(config=wd_config, entity='franamor98')

            lr = wandb.config.lr
            weight_decay = wandb.config.weight_decay

    for args in [
            {
                'model_type': model_type,
                'num_layers': num_layers,
                'batch_size': batch_size,
                'hidden_dim': hidden_dim,
                'epochs': epochs,
                'opt': opt,
                'opt_scheduler': opt_scheduler,
                'opt_restart': opt_restart,
                'loss': loss,
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
                "wandb_log": wandb_log
            },
        ]:
            args = objectview(args)
    print(args.lr,args.weight_decay)
    
    #To ensure reproducibility the best we can, here we control the sources of
    #randomness by seeding the various random number generators used in this Colab
    #For more information, see: https://pytorch.org/docs/stable/notes/randomness.html
    # Seed random number generators
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
            dataset.append(torch.load(f))
    
    """
    dataset = torch.load("../data_graphs/graph_list.pt")
    print(len(dataset))
    try:
        dataset = dataset[:args.train_size+args.val_size]
    except:
        print('Dataset is smaller than train_size + val_size')
    """
  
    print(len(dataset))

    # Create checkpoint directory if it does not exist
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    
    # Compute stats
    stats_list = compute_stats_batch(dataset)

    #create the model
    num_node_features = dataset[0].x.shape[1]
    num_edge_features = dataset[0].edge_attr.shape[1]
    num_classes = 2 # the dynamic variables have the shape of 2 (velocity)

    if args.model_type == 'meshgraphnet':
        model = MeshGraphNet(num_node_features, num_edge_features, args.hidden_dim, num_classes,
                            args)
    if args.model_type == 'gunet':
        model = GUNet(num_node_features, num_classes)
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


