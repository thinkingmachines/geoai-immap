# Detection of Rapidly Growing Informal Settlements 

This repository accompanies our research work for Informal Settlement Detection in Northern Colombia.

The goal of this project is to provide a means for faster, cheaper, and more scalable detection of rapidly growing informal settlements using low-resolution satellite images and machine learning.

## Code Organization
This repository is divided into three main parts:

- **notebooks/**: contains all Jupyter notebooks for data processing and modeling
- **data/**: contains the informal settlement datasets; also the destination for downloaded satellite imagery

## Sentinel-2 Imagery
The scripts for downloading and pre-processing of low-resolutions Sentinel-2 satellite images can be found in `notebooks/01_Data_Preprocessing.ipynb`. Note that you will need to download and install the [Sentinel-2 Toolbox Sen2Cor processor](http://step.esa.int/main/third-party-plugins-2/sen2cor/sen2cor_v2-8/).

To install Sen2Cor-2.08.00, run:
```sh
wget http://step.esa.int/thirdparties/sen2cor/2.8.0/Sen2Cor-02.08.00-Linux64.run
chmod +x Sen2Cor-02.08.00-Linux64.run
./Sen2Cor-02.08.00-Linux64.run
```
More information on how to use the Sen2Cor tool can be found in the notebook.

## Acknowledgments
This work is supported by the [iMMAP Colombia](https://immap.org/colombia/).

