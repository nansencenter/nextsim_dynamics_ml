
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
from torch_geometric.loader import DataLoader


from models.GCNN_node import GCNN_node
from models.training_utils import process_dataloader
from datasets.Ice_graph_dataset import Ice_graph_dataset
from ice_graph.ice_graph import Ice_graph
from utils.graph_utils import standardize_graph_features


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
            print(f"Epoch {epoch+1}/{num_epochs}, Average Training Loss: {training_loss:.4f}")

        val_loss = process_dataloader(model, val_dataloader, device, criterion=loss)
        validation_losses.append(val_loss)

        if epoch % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Average Validation Loss: {val_loss:.4f}")

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

    #usefull to avoid local minimas
    #scheduler = ReduceLROnPlateau(optimizer, 'min')
    time_index = 1#index of element graph to fetch samples

    n_generations = 4000
    radius = 600000 #meters
    iterations = 10
    n_neighbours = 4
    val_time_index = time_index+iterations
    #get train and val samples around a central point
    samples_train = nextsim.get_samples_area((0,0),radius,time_index=time_index,n_samples=n_generations)
    samples_val = nextsim.get_samples_area((0,0),radius,time_index=val_time_index,n_samples=int(n_generations/4))

   
    train_graph_list = []
    for sample in tqdm(samples_train,"Generating training graphs"):
        graph = nextsim.get_element_graph(sample,time_index=time_index,n_neighbours=n_neighbours,target_iter=iterations,predict_element=True)
        if graph is not None:
            train_graph_list.append(graph)

    val_graph_list = []
    for sample in tqdm(samples_val,"Generating validation graphs"):
        graph = nextsim.get_element_graph(sample,time_index=val_time_index,n_neighbours=n_neighbours,target_iter=iterations,predict_element=True)
        if graph is not None:
            val_graph_list.append(graph)

    #standardize the features
    train_graph_list = standardize_graph_features(train_graph_list)
    val_graph_list = standardize_graph_features(val_graph_list)
    #create the dataset        
    train_dataset = Ice_graph_dataset(train_graph_list, transform=None)
    val_dataset = Ice_graph_dataset(val_graph_list, transform=None)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)


    example_graph = next(iter(train_dataset)) #just to get the num_features
    num_features = example_graph.x.shape[-1]  # Node feature dimension
    hidden_channels = 8
    num_classes = example_graph.y[0].shape[0] 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCNN_node(num_features, hidden_channels, num_classes,dropout=0.3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=5e-4)
    loss = nn.MSELoss()
    num_epochs = 20

    scheduler = ExponentialLR(optimizer, gamma=0.9)

    # Define the number of epochs
    train_model(model, optimizer, scheduler, num_epochs, device, loss, train_dataloader,val_dataloader)


if __name__ == "__main__":
    main()