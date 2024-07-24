"""对scdata和stdata数据进行过滤"""
import os

import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib import pyplot as plt

import MACD.simulation_st as simulation_st
import anndata

from scipy.sparse import issparse,csr_matrix
from sklearn.preprocessing import normalize
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects

from MACD import data_downsample


def normalized_data(ad,target_sum=None):
    ad_norm = sc.pp.normalize_total(ad,inplace=False,target_sum=1e4)
    ad_norm  = sc.pp.log1p(ad_norm['X'])
    # ad_norm  = sc.pp.scale(ad_norm)
    # ad_norm = normalize(ad_norm,axis=1)
    ad.layers['norm'] = ad_norm
    return ad
def filter_model_genes(
    sc_ad,
    st_ad,
    celltype_key,
    layer='norm',
    deg_method=None,
    log2fc_min=0.5,
    pval_cutoff=0.01,
    n_top_markers=500,
    n_top_hvg=None,
    pct_diff=None,
    pct_min=0.1,
):
    # Remove duplicate genes from st_ad
    if len(set(st_ad.var_names)) != len(st_ad.var_names):
        print("Removing duplicate genes from st_ad")
        # Create a boolean mask where True indicates the first occurrence of each gene
        mask = ~st_ad.var_names.duplicated()
        st_ad = st_ad[:, mask].copy()

    # Compute overlapping genes
    overlaped_genes = np.intersect1d(sc_ad.var_names, st_ad.var_names)

    sc_ad = sc_ad[:,overlaped_genes].copy()
    st_ad = st_ad[:,overlaped_genes].copy()

    if n_top_hvg is None:
        st_genes = st_ad.var_names
    else:
        sc.pp.highly_variable_genes(st_ad, n_top_genes=n_top_hvg, flavor='seurat_v3')
        st_genes = st_ad.var_names[st_ad.var['highly_variable'] == True]

    sc_ad = sc_ad[:, st_genes].copy()
    sc_genes = find_sc_markers(sc_ad, celltype_key, layer, deg_method, log2fc_min, pval_cutoff, n_top_markers, pct_diff, pct_min)
    used_genes = np.intersect1d(sc_genes,st_genes)
    sc_ad = sc_ad[:,used_genes].copy()
    st_ad = st_ad[:,used_genes].copy()
    sc.pp.filter_cells(sc_ad, min_genes=1)
    sc.pp.filter_cells(st_ad,min_genes=1)
    # print(f'### This Sample Used gene numbers is: {len(used_genes)}')
    print(f'### This Sample Used gene numbers is: {len(used_genes)}')
    return sc_ad, st_ad


def find_sc_markers(sc_ad, celltype_key, layer='norm', deg_method=None, log2fc_min=0.5, pval_cutoff=0.01,
                    n_top_markers=500, pct_diff=None, pct_min=0.1):
    print('### Finding marker genes...')
    # filter celltype contain only one sample.
    filtered_celltypes = list(
        sc_ad.obs[celltype_key].value_counts()[(sc_ad.obs[celltype_key].value_counts() == 1).values].index)
    if len(filtered_celltypes) > 0:
        sc_ad = sc_ad[sc_ad.obs[~(sc_ad.obs[celltype_key].isin(filtered_celltypes))].index, :].copy()
    sc.tl.rank_genes_groups(sc_ad, groupby=celltype_key, pts=True, layer=layer, use_raw=False, method=deg_method)
    marker_genes_dfs = []
    for c in np.unique(sc_ad.obs[celltype_key]):
        tmp_marker_gene_df = sc.get.rank_genes_groups_df(sc_ad, group=c, pval_cutoff=pval_cutoff, log2fc_min=log2fc_min)
        if (tmp_marker_gene_df.empty is not True):
            tmp_marker_gene_df.index = tmp_marker_gene_df.names
            tmp_marker_gene_df.loc[:, celltype_key] = c
            if pct_diff is not None:
                pct_diff_genes = sc_ad.var_names[np.where((sc_ad.uns['rank_genes_groups']['pts'][c] -
                                                           sc_ad.uns['rank_genes_groups']['pts_rest'][c]) > pct_diff)]
                tmp_marker_gene_df = tmp_marker_gene_df.loc[np.intersect1d(pct_diff_genes, tmp_marker_gene_df.index), :]
            if pct_min is not None:
                # pct_min_genes = sc_ad.var_names[np.where((sc_ad.uns['rank_genes_groups']['pts'][c]) > pct_min)]
                tmp_marker_gene_df = tmp_marker_gene_df[tmp_marker_gene_df['pct_nz_group'] > pct_min]
            if n_top_markers is not None:
                tmp_marker_gene_df = tmp_marker_gene_df.sort_values('logfoldchanges', ascending=False)
                tmp_marker_gene_df = tmp_marker_gene_df.iloc[:n_top_markers, :]
            marker_genes_dfs.append(tmp_marker_gene_df)
    marker_gene_df = pd.concat(marker_genes_dfs, axis=0)
    print(marker_gene_df[celltype_key].value_counts())
    all_marker_genes = np.unique(marker_gene_df.names)
    return all_marker_genes

def generate_sm_stdata(sc_ad,num_sample,celltype_key,n_threads,cell_counts,clusters_mean,cells_mean,cells_min,cells_max,cell_sample_counts,cluster_sample_counts,ncell_sample_list,cluster_sample_list):
    sm_data,sm_labels = simulation_st.generate_simulation_data(sc_ad,num_sample=num_sample,celltype_key=celltype_key,downsample_fraction=None,data_augmentation=False,n_cpus=n_threads,
                                                               cell_counts=cell_counts,clusters_mean=clusters_mean,cells_mean=cells_mean,cells_min=cells_min,
                                                               cells_max=cells_max,cell_sample_counts=cell_sample_counts,cluster_sample_counts=cluster_sample_counts,
                                                               ncell_sample_list=ncell_sample_list,cluster_sample_list=cluster_sample_list)
    sm_data_mtx = csr_matrix(sm_data)
    sm_ad = anndata.AnnData(sm_data_mtx)
    sm_ad.var.index = sc_ad.var_names
    sm_labels = (sm_labels.T/sm_labels.sum(axis=1)).T
    sm_ad.obsm['label'] = pd.DataFrame(sm_labels,columns=np.array(sc_ad.obs[celltype_key].value_counts().index.values),index=sm_ad.obs_names)
    return sm_ad

def downsample_sm_spot_counts(sm_ad,st_ad,n_threads):
    fitdistrplus = importr('fitdistrplus')
    lib_sizes = robjects.FloatVector(np.array(st_ad.X.sum(1)).reshape(-1))
    res = fitdistrplus.fitdist(lib_sizes,'lnorm')
    loc = res[0][0]
    scale = res[0][1]
    sm_mtx_count = sm_ad.X.toarray()
    sample_cell_counts = np.random.lognormal(loc,scale,sm_ad.shape[0])
    sm_mtx_count_lb = data_downsample.downsample_matrix_by_cell(sm_mtx_count,sample_cell_counts.astype(np.int64), n_cpus=n_threads, numba_end=False)
    sm_ad.X = csr_matrix(sm_mtx_count_lb)


def check_data_type(ad):
    if issparse(ad.X):
        ad.X = ad.X.toarray()
    if ad.X.dtype != np.float32:
        ad.X =ad.X.astype(np.float32)
    return ad

def SaveLossPlot(SavePath, metric_logger, loss_type, output_prex):
    if not os.path.exists(SavePath):
        os.mkdir(SavePath)
    for i in range(len(loss_type)):
        plt.subplot(3, 3, i+1)
        plt.plot(metric_logger[loss_type[i]])
        plt.title(loss_type[i], x = 0.5, y = 0.5)
    imgName = os.path.join(SavePath, output_prex +'.png')
    plt.savefig(imgName)
    plt.close()