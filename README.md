# nextsim_dynamics_ml
Repository containing the main code for a ML-based model of neXTSIM dynamics.


The repository contains a main source code directory in 'src', a directory with 'notebooks' containing different jupyter notebooks to interpret or communicate ideas in a more visual way, and some other directories for data,figures... The Main repository structure is the following:

```

├── data
│   ├── ...
├── environment.yml
├── figures
│   ├── indexes
│   └── velocities
├── LICENSE
├── notebooks
│   ├── convert_nextsim_bin_files.ipynb
│   ├── example_open_nextsim_bin_files.ipynb
│   ├── example_open_nextsim_netcdf.ipynb
│   ├── tests_neXTSIM.ipynb
│   └── visualization_lab.ipynb
├── README.md
└── src
    ├── datasets
    ├── models
    └── utils
```

## Setup enviroment
To run the conda environment exported in 'environment.yml' needed, it may take a while to install.
```
conda env create -f environment.yml
conda activate nextsim_ml