# Detection of Rapidly Growing Informal Settlements 

This repository accompanies our research work for Informal Settlement Detection in Northern Colombia.

The goal of this project is to provide a means for faster, cheaper, and more scalable detection of rapidly growing informal settlements using low-resolution satellite images and machine learning.

## Requirements
* Anaconda 3
* google-earth-engine
* gdal
* pandas_ml

(for complete list see conda environment yml files)

## Setup 
* To get started, create a conda environment as follows:
	```sh
	conda env create -f {ENVIRONMENT}.yml
	```
    
	where ENVIRONMENT can be either 'processing_environment' or 'modelling_environment'. For more details, see notebooks/README.md

## Code Organization
This repository is divided into three main parts:

- **data/**: contains the informal settlement datasets; also the destination for downloaded satellite imagery
- **notebooks/**: contains all Jupyter notebooks for data processing and model experimentation
- **utils/**: contains utility scripts for geospatial data pre-processing and modeling

For details on how to add your own data, please see data/README.md. For privacy concerns, we did not include in this repo the labelled training data that identified informal settlements in Colombia. If you need this dataset, please contact ThinkingMachines or IMMAP at hello@thinkingmachin.es, info@immap.org.

## Acknowledgments
This work is supported by the [iMMAP Colombia](https://immap.org/colombia/).
