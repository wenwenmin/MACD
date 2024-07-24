"""https://github.com/QuKunLab/SpatialBenchmarking/blob/main/Codes/Deconvolution/Stereoscope_pipeline.py"""
import copy
import torch
import scanpy as sc
import numpy as np
import pandas as pd

import scvi
from scvi.external import RNAStereoscope, SpatialStereoscope

import sys

import os
def check_and_convert_to_integers(tensor):
    if not torch.all(tensor == tensor.to(torch.int)):
        print("输入数据包含浮点数，正在转换为整数...")
        tensor = tensor.to(torch.int)
    return tensor
def main(a,b,cell_key):
    for i in range(a, b):
        sc_file = 'Datasets/preproced_data\dataset' + str(i) + '\Scdata_filter.h5ad'
        st_file = 'Datasets/preproced_data\dataset' + str(i)+ '\Real_STdata_filter.h5ad'

        celltype_key = cell_key
        output_path = 'Baselines/Stereoscope\Result\dataset' + str(i)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        sc_adata1 = sc.read_h5ad(sc_file)
        st_adata1 = sc.read_h5ad(st_file)

        sc_adata = copy.deepcopy(sc_adata1)
        st_adata = copy.deepcopy(st_adata1)

        sc.pp.filter_genes(sc_adata, min_counts=10)

        non_mito_genes_list = [name for name in sc_adata.var_names if not name.startswith('MT-')]
        sc_adata = sc_adata[:, non_mito_genes_list]

        sc_adata.layers["counts"] = sc_adata.X.copy()
        sc.pp.normalize_total(sc_adata, target_sum=1e5)
        sc.pp.log1p(sc_adata)
        sc_adata.raw = sc_adata

        sc.pp.highly_variable_genes(
            sc_adata,
            n_top_genes=7000,
            subset=True,
            layer="counts",
            flavor="seurat_v3",
            span=1
        )
        if len(set(st_adata.var_names)) != len(st_adata.var_names):
            print("Removing duplicate genes from st_ad")
            # Create a boolean mask where True indicates the first occurrence of each gene
            mask = ~st_adata.var_names.duplicated()
            st_adata = st_adata[:, mask].copy()
        intersect = np.intersect1d(sc_adata.var_names, st_adata.var_names)
        st_adata = st_adata[:, intersect].copy()
        sc_adata = sc_adata[:, intersect].copy()

        # scvi.data.setup_anndata(sc_adata, layer="counts", labels_key=celltype_key)
        RNAStereoscope.setup_anndata(sc_adata, layer="counts", labels_key=celltype_key)

        print(sc_adata1)

        stereo_sc_model = RNAStereoscope(sc_adata)
        stereo_sc_model.train(max_epochs=50)
        # stereo_sc_model.history["elbo_train"][10:].plot()

        st_adata.layers["counts"] = st_adata.X.copy()
        # scvi.data.setup_anndata(st_adata, layer="counts")
        scvi.external.SpatialStereoscope.setup_anndata(st_adata, layer="counts")
        spatial_model = SpatialStereoscope.from_rna_model(st_adata, stereo_sc_model)
        spatial_model.train(max_epochs=350)
        # spatial_model.history["elbo_train"][10:].plot()
        spatial_model.get_proportions().to_csv(output_path + '/Stereoscope_result.csv')