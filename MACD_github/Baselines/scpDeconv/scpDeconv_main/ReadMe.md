# Deep Domain Adversarial Neural Network for the Deconvolution of Cell Type Mixtures in Tissue Proteome Profiling

**scpDeconv** is a novel deep learning-based deconvolution method tailored to single-cell proteomic data sets. scpDeconv uses an autoencoder to leverage the information from bulk proteomic data to improve the quality of single-cell proteomic data, and employs a domain adversarial architecture to bridge the single-cell and bulk data distributions and transfer labels from single-cell data set to bulk data set.
<p align="center">
  <img width="60%" src="https://github.com/TencentAILabHealthcare/scpDeconv/blob/main/img/img1.png">
</p>
More details can be found in paper: https://www.biorxiv.org/content/10.1101/2022.11.25.517895v2

## Setup

### Dependencies

Workflow of scpDeconv are implemented in python.


[![python >3.6.8](https://img.shields.io/badge/python-3.6.8-brightgreen)](https://www.python.org/) 

[![scipy-1.5.4](https://img.shields.io/badge/scipy-1.5.4-yellowgreen)](https://github.com/scipy/scipy) [![torch-1.10.2](https://img.shields.io/badge/torch-1.10.2-orange)](https://github.com/pytorch/pytorch) [![numpy-1.19.2](https://img.shields.io/badge/numpy-1.19.2-red)](https://github.com/numpy/numpy) [![pandas-1.1.5](https://img.shields.io/badge/pandas-1.1.5-lightgrey)](https://github.com/pandas-dev/pandas) [![scanpy-1.7.2](https://img.shields.io/badge/scanpy-1.7.2-blue)](https://github.com/theislab/scanpy) [![scikit__learn-0.24.2](https://img.shields.io/badge/scikit__learn-0.24.2-green)](https://github.com/scikit-learn/scikit-learn)

### Installation and Usage

The running environment of scpDeconv can be installed from docker-hub repository: 

Step1: Pull the docker image from docker-hub:

	> docker pull avonwang/scpdeconv:latest

Step2: Run the docker image (GPU is needed):

	> docker run --name scpdeconv --gpus all -it --rm avonwang/scpdeconv:latest /bin/bash

Step3: Download this repository:  

	> git clone https://github.com/TencentAILabHealthcare/scpDeconv.git

Step4: Run the demo task (murine_cellline) within the docker container on your computing machine:

	> cd scpDeconv/
    > python3 main.py --dataset murine_cellline

### Instuctions

- The demo files to run the deconvolution task can be found in the `data/murine_cellline` folder.   

- Source codes of scpDeconv can be found in the `model` folder. 

- Running script is `main.py`. 

- Before running scpDeconv, `options.py` can be edited according to the data used (See Parameters section for more details).

- The result files of the demo task will be stored in the `Result/murine_cellline` folder.

## Parameters

The parameters of scpDeconv are listed in `options.py` script, which need to be edited according to the data used in your deconvolution task.

### Input parameters

- `data_dir`: data path for input files (single cell proteomic data and tissue proteomic data need to stay in the same data path)



- `ref_dataset_name`: expression file name of single cell proteimic data (.h5ad or .csv format)

- `ref_metadata_name`: metadata file name of single cell proteimic data (.csv format)

- `target_dataset_name`: expression file name of tissue proteomic data (.h5ad or .csv format)

- `target_metadata_name`: metadata file name of tissue proteomic data (.csv format)

- `random_type`: column name in metadata of AnnData which lists cell type labels for single cell proteomic data



- `type_list`: cell types deconvoluted in this running task (set as `None` by default)

### Training parameters



- `ref_sample_num`: sample number of simulated reference data in stage 1 (set as 4000 by default)



- `sample_size`: sample size of simulated reference data in stage 1 (set as 15 by default)



- `HVP_num`: number of top highly variable proteins within target data (set as 500 by default)



- `target_type`: source type of target tissue proteomic data (`simulated` or `real`, set as `real` by default)



- `target_sample_num`: sample number of simulated target data in stage 1, needed when target_type is `simulated` (set as 1000 by default)



- `batch_size`: batch size for stage 2 and 3 in scpDeconv (set as 50 by default)



- `epochs`: training epoch for stage 3 in scpDeconv (set as 30 by default)



- `learning_rate`: learning rate for stage 2 and 3 in scpDeconv (set as 0.0001 by default)

### Output parameters



- `SaveResultsDir`: data path for output files

- `target_predicted_fractions.csv`: output file for predicted cell type proportions

## Data

Input single cell proteomic data are in h5ad or csv format. Target data are in h5ad or csv format. Prediction results will be saved in csv format.

### Input single cell proteomic data format

H5ad or csv files are needed for input single cell proteomic data. 

If single cell proteomic data are in h5ad format, cell metadata including celltype labels need to be stored in obs of Anndata object. The protein name or ID need to be kept in var_names of h5ad.

If single cell proteomic data are in csv format, an extra csv file with cell metadata information need to be provided. And the protein name or ID need to be kept in the column name of csv file.

### Input tissue proteomic data format

A h5ad or csv file is needed for input tissue proteomic data. The protein name or ID need to be kept in var_names of h5ad or in the column name of csv file.

### Output prediction results format

The output prediction results will be saved in csv format and named `target_predicted_fractions.csv`:

sample_ID |	cell_type1 | cell_type2 | cell_type3 | cell_type4 | cell_type5 | cell_type6
--- | --- | --- | --- | --- | --- | ---
sample1 | 0.27427006 | 0.2003956 | 0.2496997 | 0.08540553 | 0.120905206 | 0.06932399
sample2 | 0.10802831 | 0.09279254 | 0.03515872 | 0.29576084 | 0.17291941 | 0.2953402
sample3 | 0.11173443 | 0.19791068 | 0.043168757 | 0.24210045 | 0.18518583 | 0.2198999

Note:
	
1. The expression matrixs of single cell proteomic data and tissue proteomic data are protein abundance matrixs (normalized).
2. The cell type labels of single cell proteomic data are required, while the corresponding cell type proportions of tissue proteomic data are optional and will only be used in the calculation of prediction accuracy.
3. The protein name/ID of input single cell proteomic data and tissue proteomic data must keep the same format and source. 

For more details, please refer to paper.

## Time Cost

Taking demo murine_cellline data (1000 pseudo-target-samples) and default parameters as an example, typical running time on a "normal" desktop computer is about 3 minutes.

## Disclaimer

This tool is for research purpose and not approved for clinical use.

This is not an official Tencent product.

## Copyright

This tool is developed in Tencent AI Lab.

The copyright holder for this project is Tencent AI Lab.

All rights reserved.

## Citation

If you find our work helpful in your resarch or work, please cite us.

Wang, F., Yang, F., Huang, L. et al. Deep domain adversarial neural network for the deconvolution of cell type mixtures in tissue proteome profiling. Nat Mach Intell (2023). https://doi.org/10.1038/s42256-023-00737-y

## Questions

If you have any questions or problems, please feel free to open a new issue [here](https://github.com/TencentAILabHealthcare/scpDeconv/issues). We will fix the new issue ASAP. 
