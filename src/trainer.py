
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
from torch_geometric.loader import DataLoader


from models.GNNs import GCNN_2G
from models.training_utils import process_dataloader
from datasets.Ice_graph_dataset import Ice_graph_dataset
from ice_graph.ice_graph import Ice_graph
from utils.graph_utils import compute_normalization_batch


import numpy as np
from tqdm import tqdm

import os
from typing import List, Optional,Tuple



def train_model(
    model: torch.nn.Module,
    optimizer: [torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    num_epochs: int,
    device: torch.device,
    loss: torch.nn,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader
) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    training_losses = []
    validation_losses = []

    for epoch in range(num_epochs):
        training_loss = process_dataloader(model, train_dataloader, device, optimizer, scheduler, loss)
        training_losses.append(training_loss)

        if epoch % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"\tAverage Training Loss: {training_loss:.8f}")
        val_loss = process_dataloader(model, val_dataloader, device, criterion=loss)
        validation_losses.append(val_loss)

        if epoch % 5 == 0:
            print(f"\tAverage Validation Loss: {val_loss:.4f}")

        scheduler.step() if scheduler else None

    return training_losses, validation_losses


def main(
    graph_path: str = "example_data",
):
    np.random.seed(42)
    print()
    
    file_graphs = [dict(np.load(f'{graph_path}/{file}')) for file in sorted(os.listdir(graph_path)) ]
    print(f'Loaded {len(file_graphs)} graphs')

    nextsim = Ice_graph(
        file_graphs,
        vertex_element_features =
            ['M_wind_x',
            'M_wind_y',
            'M_ocean_x',
            'M_ocean_y',
            'x',
            'y']
    )
 

    n_generations = 100
    predict_vel = True

    radius = 500000 #meters
    iterations = 1
    time_index = 3 #index of element graph to fetch samples from
    time_index_val = 20

    samples_train = nextsim.get_samples_area((0,0),radius,time_index=time_index,n_samples=n_generations,elements=False)
    samples_val = nextsim.get_samples_area((0,0),radius,time_index=time_index_val,n_samples=int(n_generations/5),elements=False)


    train_graph_list = []
    fet = ['Damage', 'Concentration', 'Thickness', 'M_wind_x', 'M_wind_y', 'M_ocean_x', 'M_ocean_y', 'x', 'y']

    train_graph_list = nextsim.get_samples_graph(
        (0,0),
        radius,
        [time_index,time_index+2,time_index+4,time_index+6,time_index+8,time_index+10,time_index+12,time_index+14,time_index+16],
        n_samples=n_generations,
        target_iter=iterations,
        e_features=fet,
        include_vertex=True,
        pred_velocity=predict_vel

    )
    val_graph_list = nextsim.get_samples_graph(
        (0,0),
        radius,
        [time_index_val,time_index_val+2,time_index_val+4],
        n_samples=int(n_generations/5),
        target_iter=iterations,
        e_features=fet,
        include_vertex=True,
        pred_velocity=predict_vel

    )

    #compute norm transform
    transform_train = compute_normalization_batch(train_graph_list)
    transform_val = compute_normalization_batch(val_graph_list)

    #create datasets and loaders
    batch_size = 128
    train_dataset = Ice_graph_dataset(train_graph_list, transform=None)
    val_dataset = Ice_graph_dataset(val_graph_list, transform=None)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    e_graph = next(iter(train_dataset))[0] #just to get the num_features
    num_features1 = e_graph.x.shape[-1]  # Node feature dimension

    v_graph = next(iter(train_dataset))[1] #just to get the num_features
    num_features2 = v_graph.x.shape[-1]  # Node feature dimension

    hidden_channels = 12
    num_classes = e_graph.y[0].shape[0] # trajectory lenght *2, since we have x,y.

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCNN_2G(num_features1,num_features2, hidden_channels, num_classes,dropout=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=5e-4)
    scheduler = ExponentialLR(optimizer, gamma=0.95)
    loss = nn.L1Loss()
    num_epochs = 100

    # Define the number of epochs
    train_model(model, optimizer, scheduler, num_epochs, device, loss, train_dataloader,val_dataloader)


if __name__ == "__main__":
    main()