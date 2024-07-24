
"""https://github.com/QuKunLab/SpatialBenchmarking/blob/main/Codes/Deconvolution/DestVI_pipeline.py"""
import copy

import scanpy as sc
import numpy as np
import pandas as pd

import scvi
from scvi.model import CondSCVI, DestVI
import anndata as ad
import sys
import os
def main(a,b,cell_key):
    for i in range(a, b):
        sc_file = 'Datasets/preproced_data\dataset' + str(i) + '\Scdata_filter.h5ad'
        st_file = 'Datasets/preproced_data\dataset' + str(i) + '\Real_STdata_filter.h5ad'
        st_adata1 = ad.read_h5ad(st_file)
        sc_adata1 = ad.read_h5ad(sc_file)
        sc_adata = copy.deepcopy(sc_adata1)
        st_adata = copy.deepcopy(st_adata1)
        outpath = 'Baselines/DestVi\Result/' + 'dataset' + str(i)
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        intersect = np.intersect1d(sc_adata.var_names, st_adata.var_names)
        st_adata = st_adata[:, intersect].copy()
        sc_adata = sc_adata[:, intersect].copy()
        # let us filter some genes
        G = len(intersect)
        sc.pp.filter_genes(sc_adata, min_counts=10)

        sc_adata.layers["counts"] = sc_adata.X.copy()

        sc.pp.highly_variable_genes(
            sc_adata,
            n_top_genes=G,
            subset=True,
            layer="counts",
            flavor="seurat_v3"
        )

        sc.pp.normalize_total(sc_adata, target_sum=10e4)
        sc.pp.log1p(sc_adata)
        sc_adata.raw = sc_adata
        st_adata.layers["counts"] = st_adata.X.copy()
        sc.pp.normalize_total(st_adata, target_sum=10e4)
        sc.pp.log1p(st_adata)
        st_adata.raw = st_adata
        # filter genes to be the same on the spatial data
        intersect = np.intersect1d(sc_adata.var_names, st_adata.var_names)
        st_adata = st_adata[:, intersect].copy()
        sc_adata = sc_adata[:, intersect].copy()
        print(st_adata)
        CondSCVI.setup_anndata(sc_adata, layer="counts", labels_key=cell_key)

        sc_model = CondSCVI(sc_adata, weight_obs=True)
        sc_model.train(max_epochs=250, lr=0.0001)
        sc_model.history["elbo_train"].plot()
        # CondSCVI.setup_anndata(st_adata, layer="counts")
        # st_model = DestVI.from_rna_model(st_adata, sc_model)



        if hasattr(st_adata.X, 'toarray'):
            dense_matrix = st_adata.X.toarray()
        else:
            dense_matrix = st_adata.X

        # 确保数据类型为浮点数
        dense_matrix = dense_matrix.astype(np.float64)

        # 检查是否有NaN值
        row_sums = np.sum(dense_matrix, axis=1)

        # 创建一个布尔索引，表示哪些行不全为零
        non_zero_rows = row_sums != 0

        # 使用布尔索引过滤掉全为零的行
        stadata_filtered = st_adata[non_zero_rows, :].copy()
        filtered_file = outpath+'\Spatial.h5ad'
        stadata_filtered.write_h5ad(filtered_file)
        scvi.model.DestVI.setup_anndata(stadata_filtered)
        st_model = DestVI.from_rna_model(stadata_filtered, sc_model)
        st_model.train(max_epochs=1000)
        st_model.history["elbo_train"].plot()
        st_model.get_proportions().to_csv(outpath + '/DestVI_result.csv')
