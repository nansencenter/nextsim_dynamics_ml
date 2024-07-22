# nextsim_dynamics_ml

This Github Repository containing the main code for a ML-based model of neXTSIM dynamics.


The repository contains a main source code directory in 'src', a directory with 'notebooks' containing different jupyter notebooks to interpret or communicate ideas in a more visual way, and some other directories for data,figures... The Main repository structure is the following:

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
    ├── trainer_roll_out.py
    ├── trainer_singleparticle_old.py
    └── utils
        ├── graph_utils.py
        ├── metrics.py
        └── Tri_neighbors.py
```



## Setup enviroment
To run the conda environment exported in 'environment.yml' needed, it may take a while to install.
```
conda env create -f environment.yml
conda activate nextsim_ml