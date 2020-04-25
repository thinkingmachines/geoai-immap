# Detection of Rapidly Growing Informal Settlements 

This repository accompanies our research work for Informal Settlement Detection in Northern Colombia.

The goal of this project is to provide a means for faster, cheaper, and more scalable detection of rapidly growing informal settlements using low-resolution satellite images and machine learning.

## Setup 
To get started, create a virtual environment as follows:
```sh
python3 -m pip install --user virtualenv
python3 -m venv venv
ipython kernel install --user --name=venv
```

## Code Organization
This repository is divided into three main parts:

- **data/**: contains the informal settlement datasets; also the destination for downloaded satellite imagery
- **notebooks/**: contains all Jupyter notebooks for data processing and model experimentation
- **utils/**: contains utility scripts for geospatial data pre-processing and modeling


## Acknowledgments
This work is supported by the [iMMAP Colombia](https://immap.org/colombia/).
