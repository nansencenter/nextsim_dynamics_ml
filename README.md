# Data-Driven Modelling of Sea Ice Dynamics

<img src="https://github.com/user-attachments/assets/0ea3c7c0-c11e-4223-9ac3-2b9176f6678f" alt="Sea Ice Dynamics" width="700"/>

[Authors: Francisco Amor Roldán (NTNU, NERSC), Anton Korosv (NERSC)]

This repository contains the main code for an ML-based model of neXTSIM dynamics. It has a main source code directory in 'src', a directory with 'notebooks' containing different jupyter notebooks to interpret or communicate ideas in a more visual way, and some other directories for data,figures... The repository structure is the following:

```
├── environment.yml
├── example_data
├── LICENSE
├── notebooks
│   ├── examples
│   ├── Data_correlations.ipynb
│   ├── results.ipynb
│   └── roll_out.ipynb
├── README.md
└── src
    ├── datasets
    │   ├── create_MGN_dataset.py
    │   ├── Ice_graph_dataset.py
    ├── ice_graph
    │   ├── ice_graph.py
    ├── models
    │   ├── ConvGNNs_old.py
    │   ├── GUnet.py
    │   ├── MGN.py
    │   └── training_utils.py
    ├── sweep_config.yml
    ├── trainer_GNN.py
    ├── fine_tune_roll_out.py
    └── utils
        ├── graph_utils.py
        ├── metrics.py
        └── Tri_neighbors.py
```

### Notebooks

Contains different notebooks to explore and visualize data or ideas in a more graphical and interactive way. The example folder contains didactical examples of how to open and process data from neXtSIM outputs. `Data_correlations.ipynb` contains scripts for preliminary data analysis, `results.ipynb` contains scripts to interpret model results, and `roll_out.ipynb` contains scripts to apply models iteratively.

### Source (src)

Contains the main code developed in the project.

- **datasets**: Scripts to generate the datasets and data structures used by PyTorch Geometric to train the GNN models from neXtSIM outputs.
  - `create_MGN_dataset.py`
  - `Ice_graph_dataset.py`
- **ice_graph**: Contains the main code to handle and process neXtSIM output data in an object-oriented fashion. Methods to load, interpolate, find neighbors, and other utilities are implemented as methods.
  - `ice_graph.py`
- **models**: Implementation of the different GNN models used in the project alongside some training utilities.
  - `ConvGNNs_old.py`: Deprecated code for sea ice floe-based predictions used in a previous approach of this project.
  - `GUnet.py`: Code for Graph-Unets.
  - `MGN.py`: Code for the MeshGraphNet model.
  - `training_utils.py`: Scripts used during training.
- **utils**: General utilities used in the project.
  - `graph_utils.py`: Functions for graph data handling and normalization.
  - `metrics.py`: Functions for the metrics used.
  - `Tri_neighbors.py`: Class to work with node-level neighborhoods in the neXtSIM mesh.
- **trainer_GNN.py**: Code used for training the GNN models.
- **fine_tune_roll_out.py**: Code used for fine-tuning GNN models during roll-out.
- **sweep_config.yml**: Parameters used by Weights and Biases for hyperparameter optimization.

## Setup Environment

To run the conda environment exported in `environment.yml`, it may take a while to install.

```bash
conda env create -f environment.yml
conda activate nextsim_ml
```
