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
from utils.graph_utils import compute_stats_batch
from utils.graph_utils import unnormalize

def train(dataset, device, stats_list, args):
    '''
    Performs a training loop on the dataset for MeshGraphNets. Also calls
    test and validation functions.
    '''

    df = pd.DataFrame(columns=['epoch','train_loss','test_loss', 'velo_val_loss'])

    #Define the model name for saving
    model_name='model_nl'+str(args.num_layers)+'_bs'+str(args.batch_size) + \
               '_hd'+str(args.hidden_dim)+'_ep'+str(args.epochs)+'_wd'+str(args.weight_decay) + \
               '_lr'+str(args.lr)+'_shuff_'+str(args.shuffle)+'_tr'+str(args.train_size)+'_te'+str(args.test_size)

    #torch_geometric DataLoaders are used for handling the data of lists of graphs
    loader = DataLoader(dataset[:args.train_size], batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(dataset[args.train_size:], batch_size=args.batch_size, shuffle=False)

    #The statistics of the data are decomposed
    [mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge,mean_vec_y,std_vec_y] = stats_list
    (mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge,mean_vec_y,std_vec_y)=(mean_vec_x.to(device),
        std_vec_x.to(device),mean_vec_edge.to(device),std_vec_edge.to(device),mean_vec_y.to(device),std_vec_y.to(device))

    # build model
    num_node_features = dataset[0].x.shape[1]
    num_edge_features = dataset[0].edge_attr.shape[1]
    num_classes = 2 # the dynamic variables have the shape of 2 (velocity)

    model = MeshGraphNet(num_node_features, num_edge_features, args.hidden_dim, num_classes,
                            args).to(device)
    scheduler, opt = build_optimizer(args, model.parameters())

    # train
    losses = []
    test_losses = []
    velo_val_losses = []
    best_test_loss = np.inf
    best_model = None
    for epoch in trange(args.epochs, desc="Training", unit="Epochs"):
        total_loss = 0
        model.train()
        num_loops=0
        for batch in loader:
            #Note that normalization must be done before it's called. The unnormalized
            #data needs to be preserved in order to correctly calculate the loss
            batch=batch.to(device)
            opt.zero_grad()         #zero gradients each time
            pred = model(batch,mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge)
            loss = model.loss(pred,batch,mean_vec_y,std_vec_y)
            loss.backward()         #backpropagate loss
            opt.step()
            total_loss += loss.item()
            num_loops+=1
        total_loss /= num_loops
        losses.append(total_loss)

        #Every tenth epoch, calculate acceleration test loss and velocity validation loss
        if epoch % 10 == 0:
            if (args.save_velo_val):
                # save velocity evaluation
                test_loss, velo_val_rmse = test(test_loader,device,model,mean_vec_x,std_vec_x,mean_vec_edge,
                                 std_vec_edge,mean_vec_y,std_vec_y, args.save_velo_val)
                velo_val_losses.append(velo_val_rmse.item())
            else:
                test_loss, _ = test(test_loader,device,model,mean_vec_x,std_vec_x,mean_vec_edge,
                                 std_vec_edge,mean_vec_y,std_vec_y, args.save_velo_val)

            test_losses.append(test_loss.item())

            # saving model
            if not os.path.isdir( args.checkpoint_dir ):
                os.mkdir(args.checkpoint_dir)

            PATH = os.path.join(args.checkpoint_dir, model_name+'.csv')
            df.to_csv(PATH,index=False)

            #save the model if the current one is better than the previous best
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_model = copy.deepcopy(model)

        else:
            #If not the tenth epoch, append the previously calculated loss to the
            #list in order to be able to plot it on the same plot as the training losses
            if (args.save_velo_val):
              test_losses.append(test_losses[-1])
              velo_val_losses.append(velo_val_losses[-1])

        if (args.save_velo_val):
            #concat dict to existinf dataframe
            df = pd.concat([df, pd.DataFrame({'epoch': [epoch], 'train_loss': losses[-1:], 'test_loss': test_losses[-1:],'velo_val_loss': velo_val_losses[-1:]})])
            """df.append({'epoch': epoch,'train_loss': losses[-1],
                            'test_loss':test_losses[-1],
                           'velo_val_loss': velo_val_losses[-1]}, ignore_index=True)
            """
        else:
            df = pd.concat([df, pd.DataFrame({'epoch': [epoch], 'train_loss': losses[-1:], 'test_loss': test_losses[-1:]})] )
            #df = df.append({'epoch': epoch, 'train_loss': losses[-1], 'test_loss': test_losses[-1]}, ignore_index=True)
        if(epoch%100==0):
            if (args.save_velo_val):
                print("train loss", str(round(total_loss, 2)),
                      "test loss", str(round(test_loss.item(), 2)),
                      "velo loss", str(round(velo_val_rmse.item(), 5)))
            else:
                print("train loss", str(round(total_loss,2)), "test loss", str(round(test_loss.item(),2)))


            if(args.save_best_model):

                PATH = os.path.join(args.checkpoint_dir, model_name+'.pt')
                torch.save(best_model.state_dict(), PATH )

    return test_losses, losses, velo_val_losses, best_model, best_test_loss, test_loader,model_name

def test(loader,device,test_model,
         mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge,mean_vec_y,std_vec_y, is_validation,
          delta_t=0.01, save_model_preds=False, model_type=None):

    '''
    Calculates test set losses and validation set errors.
    '''

    loss=0
    velo_rmse = 0
    num_loops=0

    for data in loader:
        data=data.to(device)
        with torch.no_grad():

            #calculate the loss for the model given the test set
            pred = test_model(data,mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge)
            loss += test_model.loss(pred, data,mean_vec_y,std_vec_y)

            #calculate validation error if asked to
            if (False):##

                #Like for the MeshGraphNets model, calculate the mask over which we calculate
                #flow loss and add this calculated RMSE value to our val error
                normal = torch.tensor(0)
                outflow = torch.tensor(5)
                #loss_mask = torch.logical_or((torch.argmax(data.x[:, 2:], dim=1) == torch.tensor(0)),
                                             #(torch.argmax(data.x[:, 2:], dim=1) == torch.tensor(5)))

                eval_velo = data.x[:, 0:2] + unnormalize( pred[:], mean_vec_y, std_vec_y ) * delta_t
                gs_velo = data.x[:, 0:2] + data.y[:] * delta_t

                error = torch.sum((eval_velo - gs_velo) ** 2, axis=1)
                #velo_rmse += torch.sqrt(torch.mean(error[loss_mask]))
                velo_rmse += torch.sqrt(torch.mean(error))


        num_loops+=1
        # if velocity is evaluated, return velo_rmse as 0
    return loss/num_loops, velo_rmse/num_loops

def build_optimizer(args, params):
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p : p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.opt_restart)
    return scheduler, optimizer


class objectview(object):
    def __init__(self, d):
        self.__dict__ = d



def main(
    dataset_dir='../data_graphs/graph_list.pt',
    model_type='meshgraphnet',
    num_layers=6,
    batch_size=16,
    hidden_dim=4,
    epochs=100,
    opt='adam',
    opt_scheduler='cos',
    opt_restart=0,
    weight_decay=5e-4,
    lr=0.001,
    train_size=5,
    test_size=2,
    device='cpu',
    shuffle=True,
    save_velo_val=False,
    save_best_model=True,
    checkpoint_dir='../best_models/',
    postprocess_dir='./2d_loss_plots/'
):

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
                'weight_decay': weight_decay,
                'lr': lr,
                'train_size': train_size,
                'test_size': test_size,
                'device': device,
                'shuffle': shuffle,
                'save_velo_val': save_velo_val,
                'save_best_model': save_best_model,
                'checkpoint_dir': checkpoint_dir,
                'postprocess_dir': postprocess_dir,
            },
        ]:
            args = objectview(args)

    #To ensure reproducibility the best we can, here we control the sources of
    #randomness by seeding the various random number generators used in this Colab
    #For more information, see: https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(5)  #Torch
    random.seed(5)        #Python
    np.random.seed(5)     #NumPy

    #load the dataset
    dataset = torch.load(dataset_dir)
    try:
        dataset = dataset[:args.train_size+args.test_size]
    except:
        print('Dataset is smaller than train_size + test_size')

    if(args.shuffle):
        random.shuffle(dataset)

    #create best_models directory if it does not exist
    if not os.path.isdir( args.checkpoint_dir ):
        os.mkdir(args.checkpoint_dir)
    
    stats_list = compute_stats_batch(dataset)

    test_losses, losses, velo_val_losses, best_model, best_test_loss, test_loader,model_name = train(dataset, device, stats_list, args)

    print("Min test set loss: {0}".format(min(test_losses)))
    print("Minimum loss: {0}".format(min(losses)))
    print("Best model: {0}".format(model_name))
    if (args.save_velo_val):
        print("Minimum velocity validation loss: {0}".format(min(velo_val_losses)))
    

if __name__ == "__main__":
    main()