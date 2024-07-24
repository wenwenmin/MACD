import os

import numpy as np
import torch
from scipy.sparse import csr_matrix
import numba
import logging
import random
import MACD.simulation_st
from MACD.utils import *
import scvi

def get_scvi_latent(
        st_ad,
        sm_ad,
        n_layers,
        n_latent,
        gene_likelihood='zinb',
        dispersion='gene-batch',
        max_epochs=1000,
        early_stopping=True,
        batch_size=4096,
):


    st_ad.obs["batch"] = 'real'
    sm_ad.obs["batch"] = 'simulated'

    adata = sc.concat([st_ad,sm_ad])
    adata.layers["counts"] = adata.X.copy()

    scvi.model.SCVI.setup_anndata(
        adata,
        layer="counts",
        batch_key="batch"
    )

    vae = scvi.model.SCVI(adata, n_layers=n_layers, n_latent=n_latent, gene_likelihood=gene_likelihood,
                          dispersion=dispersion)
    vae.train(max_epochs=max_epochs, early_stopping=early_stopping, batch_size=batch_size, use_gpu=True)
    adata.obsm["X_scVI"] = vae.get_latent_representation()
    # print("现在的adata.shape", adata.shape)
    st_scvi_ad = anndata.AnnData(adata[adata.obs['batch'] != 'simulated'].obsm["X_scVI"])
    sm_scvi_ad = anndata.AnnData(adata[adata.obs['batch'] == 'simulated'].obsm["X_scVI"])

    st_scvi_ad.obs = st_ad.obs
    st_scvi_ad.obsm = st_ad.obsm

    sm_scvi_ad.obs = sm_ad.obs
    sm_scvi_ad.obsm = sm_ad.obsm

    sm_scvi_ad = check_data_type(sm_scvi_ad)
    st_scvi_ad = check_data_type(st_scvi_ad)

    sm_data = sm_scvi_ad.X
    sm_labels = sm_scvi_ad.obsm['label'].values
    st_data = st_scvi_ad.X
    return sm_scvi_ad, st_scvi_ad

    return sm_scvi_ad, st_scvi_ad
def data_prepare(
    sc_ad,
    st_ad,
    celltype_key,
    h5ad_file_path,
    data_file_path,
    n_layers,
    n_latent,
    deg_method:str='wilcoxon',
    n_top_markers:int=200,
    n_top_hvg:int=None,
    log2fc_min=0.5,
    pval_cutoff=0.01,
    pct_diff=None,
    pct_min=0.1,
    sm_size:int=10000,
    cell_counts=None,
    clusters_mean=None,
    cells_mean=10,
    cells_min=1,
    cells_max=20,
    cell_sample_counts=None,
    cluster_sample_counts=None,
    ncell_sample_list=None,
    cluster_sample_list=None,
    n_threads=4,
        #seed=42,82
    seed=42,

):
    print('Setting global seed:', seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    MACD.simulation_st.numba_set_seed(seed)
    numba.set_num_threads(n_threads)

    sc_ad = normalized_data(sc_ad, target_sum=1e4)
    st_ad = normalized_data(st_ad, target_sum=1e4)
    print("原始st的形状：",st_ad.shape)
    sc_ad, st_ad = filter_model_genes(
        sc_ad,
        st_ad,
        celltype_key=celltype_key,
        deg_method=deg_method,
        n_top_markers=n_top_markers,
        n_top_hvg=n_top_hvg,
        log2fc_min=log2fc_min,
        pval_cutoff=pval_cutoff,
        pct_diff=pct_diff,
        pct_min=pct_min
    )
    print("筛选后st的形状：", st_ad.shape)
    """产生模拟数据并进行下采样"""
    sm_ad =generate_sm_stdata(sc_ad,num_sample=sm_size,celltype_key=celltype_key,n_threads=n_threads,cell_counts=cell_counts,clusters_mean=clusters_mean,cells_mean=cells_mean,
                             cells_min=cells_min,cells_max=cells_max,cell_sample_counts=cell_sample_counts,cluster_sample_counts=cluster_sample_counts,
                             ncell_sample_list=ncell_sample_list,cluster_sample_list=cluster_sample_list)

    downsample_sm_spot_counts(sm_ad,st_ad,n_threads=n_threads)
    os.makedirs(h5ad_file_path, exist_ok=True)
    sc_adcopy = sc_ad
    if 'cluster_p_balance' in sc_adcopy.uns:
        del sc_adcopy.uns['cluster_p_balance']
        del sc_adcopy.uns['cluster_p_sqrt']
        del sc_adcopy.uns['cluster_p_unbalance']
    """将scdata和stdata进行预处理后存储，以便其他Baselines使用"""
    sc_adcopy.write_h5ad(data_file_path + '\Scdata_filter.h5ad')
    sm_ad.write_h5ad(data_file_path + '\Sm_STdata_filter.h5ad')
    st_ad.write_h5ad(data_file_path + '\Real_STdata_filter.h5ad')
    """对scdata和stdata利用scvi工具进行降维"""
    sm_scvi_ad, st_scvi_ad = get_scvi_latent(st_ad,sm_ad,n_layers,n_latent)
    sm_scvi_ad.write_h5ad(h5ad_file_path + '\sm_scvi_ad.h5ad')
    st_scvi_ad.write_h5ad(h5ad_file_path + '\st_scvi_ad.h5ad')


