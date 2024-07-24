import copy
import os
import sys

from Baselines.scpDeconv.scpDeconv_main.model.DANN_model import *
from Baselines.scpDeconv.scpDeconv_main.model.utils import *
import anndata
from collections import Counter
def main(a,b,cell_key):
	for i in range(a,b):
		sc_file = 'Datasets/preproced_data\dataset' + str(i) + '\Scdata_filter.h5ad'
		sm_labelfile = 'Datasets/preproced_data\dataset' + str(i) + '\Sm_STdata_filter.h5ad'
		st_datafile = 'Datasets/preproced_data\dataset' + str(i) + '\Real_STdata_filter.h5ad'


		st_adata1 = anndata.read_h5ad(st_datafile)
		real_sc_adata1 = anndata.read_h5ad(sc_file)
		sm_labelad1 = anndata.read_h5ad(sm_labelfile)
		"""copy:保证数据不改变"""
		st_adata=copy.deepcopy(st_adata1)
		real_sc_adata = copy.deepcopy(real_sc_adata1)
		sm_labelad = copy.deepcopy(sm_labelad1)
		sc_adata = copy.deepcopy(sm_labelad1)
		# sm_data=pd.DataFrame(data=sc_adata.X.toarray(),columns=sc_adata.var_names)
		sm_data = pd.DataFrame(data=sc_adata.X.toarray(), columns=sc_adata.var_names)
		sm_lable = sm_labelad.obsm['label']

		st_data = pd.DataFrame(data=st_adata.X, columns=st_adata.var_names, index=st_adata.obs_names)


		count_ct_dict = Counter(list(real_sc_adata.obs[cell_key]))
		celltypenum = len(count_ct_dict)
		"""保存途径"""
		outfile = 'Baselines/scpDeconv\Result' + '/dataset' + str(i) + '/'
		if not os.path.exists(outfile):
			os.makedirs(outfile)
		### Run Stage 3 ###
		print("------Start Running Stage 3 : Training DANN model------")
		model_da = DANN( celltype_num=celltypenum,outdirfile=outfile)
		print(st_data.shape,sm_data.shape,sm_lable.shape)
		model_da.train(st_data=st_data, sm_data=sm_data, sm_label=sm_lable)
		print("Stage 3 : DANN model training finished!")

		### Run Stage 4 ###
		print("------Start Running Stage 4 : Inference for target data------")

		final_preds_target = model_da.prediction()
		# print(final_preds_target)
		final_preds_target.to_csv(outfile+ '/final_preds1.csv')
		final_preds_target.columns = sm_lable.columns.tolist()
		pd.DataFrame(data=final_preds_target).to_csv(outfile + '/final_preds.csv')
		print("Stage 4 : Inference for target data finished!")
