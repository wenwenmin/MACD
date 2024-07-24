"""https://github.com/QuKunLab/SpatialBenchmarking/blob/main/Codes/Deconvolution/Tangram_pipeline.py#L9"""
import copy
import os

import tangram
import anndata as ad
import anndata
import pandas as pd
import scanpy as sc
import numpy as np
import tangram as tg

def main(a,b,cell_key):
    for i in range(a, b):
        sc_file = 'Datasets/preproced_data\dataset' + str(i) + '\Scdata_filter.h5ad'
        st_file = 'Datasets/preproced_data\dataset' + str(i)+ '\Real_STdata_filter.h5ad'
        output_file_path='Baselines/Tangram\Result/' + 'dataset' + str(i)
        if not os.path.exists(output_file_path):
            os.makedirs(output_file_path)

        celltype_key = cell_key
        ad_sc1 = sc.read_h5ad(sc_file)
        ad_sp1 = sc.read_h5ad(st_file)
        ad_sc = copy.deepcopy(ad_sc1)
        ad_sp = copy.deepcopy(ad_sp1)
        # use raw count both of scrna and spatial
        sc.pp.normalize_total(ad_sc)
        celltype_counts = ad_sc.obs[celltype_key].value_counts()
        celltype_drop = celltype_counts.index[celltype_counts < 2]
        print(f'Drop celltype {list(celltype_drop)} contain less 2 sample')
        ad_sc = ad_sc[~ad_sc.obs[celltype_key].isin(celltype_drop),].copy()
        sc.tl.rank_genes_groups(ad_sc, groupby=celltype_key, use_raw=False)
        markers_df = pd.DataFrame(ad_sc.uns["rank_genes_groups"]["names"]).iloc[0:300, :]
        # print(markers_df)
        genes_sc = np.unique(markers_df.melt().value.values)
        # print(genes_sc)
        genes_st = ad_sp.var_names.values
        genes = list(set(genes_sc).intersection(set(genes_st)))
        tg.pp_adatas(ad_sc, ad_sp, genes=genes)
        ad_map = tg.map_cells_to_space(
            ad_sc,
            ad_sp,
            num_epochs=800,
            mode='clusters',
            cluster_label=celltype_key)
        tg.project_cell_annotations(ad_map, ad_sp, annotation=celltype_key)
        celltype_density = ad_sp.obsm['tangram_ct_pred']
        celltype_density = (celltype_density.T / celltype_density.sum(axis=1)).T

        celltype_density.to_csv(output_file_path + '/Tangram_result.csv')




