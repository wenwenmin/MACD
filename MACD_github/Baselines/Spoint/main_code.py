"""读取数据"""
import copy
import multiprocessing as mp
from Baselines.Spoint.model import init_model
import anndata as ad
import torch
import os
import Baselines.Spoint.data_utils
sm_model='Result'

def main(a,b,cell_key):
    for i in range(a, b):
        print('第'+str(i)+'个切片')

        sc_file = 'D:\Python_Projects\MACD_github\Datasets\Simulated_datasets\dataset4\scRNA.h5ad'
        st_file = 'D:\Python_Projects\MACD_github\Datasets\Simulated_datasets\dataset4\Spatial.h5ad'
        st_ad1 = ad.read_h5ad(st_file)
        sc_data1 = ad.read_h5ad(sc_file)
        st_ad = copy.deepcopy(st_ad1)
        sc_data = copy.deepcopy(sc_data1)
        output_path = 'Baselines/Spoint\Result\dataset' + str(i)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        model = init_model(sc_ad=sc_data,
                                   st_ad=st_ad,
                                   celltype_key=cell_key,
                                   n_top_markers=500,
                                   n_top_hvg=3000)
        # model.train
        model.model_train(sm_lr=0.01,
                    st_lr=0.01)

        pre = model.deconv_spatial()

        pre.to_csv(output_path + "/proportion.csv")


# if __name__ == '__main__':
#     main(4,5,'celltype_final')
