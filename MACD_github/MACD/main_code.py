from collections import Counter

import anndata
import anndata as ad
import pandas as pd

from MACD.MCCD_model import MDCD
from MACD.data_prepare import *
import copy
def main(a,b,cell_key):
    for i in range(a, b):
        st_file = 'Datasets/Simulated_datasets\dataset' + str(i) + '\Spatial.h5ad'
        sc_file = 'Datasets/Simulated_datasets\dataset' + str(i) + '\scRNA.h5ad'
        st_data1 = ad.read_h5ad(st_file)
        sc_data1 = ad.read_h5ad(sc_file)
        sc_data=copy.deepcopy(sc_data1)
        st_data=copy.deepcopy(st_data1)
        outfile = 'MACD\Result\dataset' + str(i)
        datafile='Datasets/preproced_data\dataset' + str(i)
        if not os.path.exists(outfile):
            os.makedirs(outfile)
        if not os.path.exists(datafile):
            os.makedirs(datafile)
        """数据处理"""
        data_prepare(sc_ad=sc_data,st_ad=st_data,celltype_key=cell_key,h5ad_file_path=outfile,data_file_path=datafile,n_layers=2,n_latent=2048)
        sc_adata = anndata.read_h5ad(outfile + '\sm_scvi_ad.h5ad')
        st_adata = anndata.read_h5ad(outfile + '\st_scvi_ad.h5ad')
        real_sc_adata = anndata.read_h5ad(datafile + '\Scdata_filter.h5ad')
        sm_labelad = anndata.read_h5ad(datafile + '\Sm_STdata_filter.h5ad')
        # sm_data=pd.DataFrame(data=sc_adata.X.toarray(),columns=sc_adata.var_names)
        sm_data = pd.DataFrame(data=sc_adata.X, columns=sc_adata.var_names)
        sm_lable = sm_labelad.obsm['label']
        st_data = pd.DataFrame(data=st_adata.X, columns=st_adata.var_names)
        print(st_data.shape,sm_data.shape,sm_lable.shape)
        count_ct_dict = Counter(list(real_sc_adata.obs[cell_key]))
        celltypenum = len(count_ct_dict)


        print("------Start Running Stage------")
        model_da = MDCD(celltypenum, outdirfile=outfile, used_features=list(sm_data.columns),num_epochs=200)
        # model_da.d
        model_da.double_train(sm_data=sm_data, st_data=st_data, sm_label=sm_lable)
        final_preds_target = model_da.prediction()
        final_preds_target.to_csv(outfile + '/final_pro1.csv')
        final_preds_target.columns = sm_lable.columns.tolist()
        pd.DataFrame(data=final_preds_target).to_csv(outfile + '/final_pro.csv')
