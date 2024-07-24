import os
import sys
import pandas as pd
import anndata as ad
import numpy as np
import scipy
import scanpy as sc
import warnings
warnings.filterwarnings('ignore')

from Baselines.scpDeconv.scpDeconv_main.model.utils import *

class ReferMixup(object):
    def __init__(self, option_list):  
        self.data_path = option_list['data_dir']
        self.ref_dataset_name = option_list['ref_dataset_name']
        self.ref_metadata_name = option_list['ref_metadata_name']
        self.target_dataset_name = option_list['target_dataset_name']
        self.target_metadata_name = option_list['target_metadata_name']
        self.random_type = option_list['random_type']
        self.type_list = option_list['type_list']
        self.train_sample_num = option_list['ref_sample_num']
        self.sample_size = option_list['sample_size']
        self.HVP_num = option_list['HVP_num']
        self.target_type = option_list['target_type']
        self.target_sample_num = option_list['target_sample_num']
        self.outdir = option_list['SaveResultsDir']
        self.normalize = 'min_max'

    def mixup(self):

        # mixup reference datasets and simulate pseudo-bulk train data
        train_data_x, train_data_y = self.mixup_dataset(self.ref_dataset_name, self.ref_metadata_name, self.train_sample_num)

        # mixup to simulate pseudo target data or get real target data
        if self.target_type == "simulated":
            target_data_x, target_data_y = self.mixup_dataset(self.target_dataset_name, self.target_metadata_name, self.target_sample_num)
            target_data = ad.AnnData(X=target_data_x.to_numpy(), obs=target_data_y)
            target_data.var_names = target_data_x.columns

        elif self.target_type == "real":
            target_data = self.load_real_data(self.target_dataset_name)

        # find protein list as used features by integrating train and target 
        used_features = self.align_features(train_data_x, target_data)

        # prepare train data and target data with aligned features
        train_data = self.align_dataset(train_data_x, train_data_y, used_features)
        target_data = target_data[:,used_features]

        # SavetSNEPlot(self.outdir, train_data, output_prex='Pseudo_Bulk_Source_'+str(self.train_sample_num))
        # SavetSNEPlot(self.outdir, target_data, output_prex='Pseudo_Bulk_Target_'+str(self.target_sample_num))
            
        return train_data, target_data

    def align_features(self, train_data_x, target_data):

        used_features = set(train_data_x.columns.tolist()).intersection(set(target_data.var_names.tolist())) # overlapped features between reference and target
        
        if self.HVP_num == 0:
            used_features = list(used_features)

        elif self.HVP_num > 0:
            sc.pp.highly_variable_genes(target_data, n_top_genes=self.HVP_num)
            HVPs = set(target_data.var[target_data.var.highly_variable].index)
            used_features = list(used_features.union(HVPs))

        return used_features

    def align_dataset(self, sim_data_x, sim_data_y, used_features):

        missing_features = [feature for feature in used_features if feature not in list(sim_data_x.columns)]   

        if len(missing_features) > 0:
            missing_data_x = pd.DataFrame(np.zeros((sim_data_x.shape[0],len(missing_features))), columns=missing_features, index=sim_data_x.index)
            sim_data_x = pd.concat([sim_data_x, missing_data_x], axis=1)

        sim_data_x = sim_data_x[used_features]
        
        sim_data = ad.AnnData(
            X=sim_data_x.to_numpy(),
            obs=sim_data_y
        )
        sim_data.uns["cell_types"] = self.type_list
        sim_data.var_names = used_features

        return sim_data

    def mixup_dataset(self, dataset, metadata, sample_num):

        sim_data_x = []
        sim_data_y = []

        ref_data_x, ref_data_y = self.load_ref_dataset(dataset, metadata)

        for i in range(int(sample_num)):
            sample, label = self.mixup_cells(ref_data_x, ref_data_y, self.type_list)
            sim_data_x.append(sample)
            sim_data_y.append(label)

        sim_data_x = pd.concat(sim_data_x, axis=1).T
        sim_data_y = pd.DataFrame(sim_data_y, columns=self.type_list)

        # Scale pseudo-bulk data
        if self.normalize:
            sim_data_x_scale = sample_normalize(sim_data_x, normalize_method=self.normalize)
            sim_data_x_scale = pd.DataFrame(sim_data_x_scale, columns=sim_data_x.columns)
            sim_data_x = sim_data_x_scale

        return sim_data_x, sim_data_y

    def load_ref_dataset(self, dataset, metadata):
        
        if ".h5ad" in dataset:
            filename = os.path.join(self.data_path, dataset)

            try:
                data_h5ad = ad.read_h5ad(filename)
                # Extract celltypes
                if self.type_list == None:
                    self.type_list = list(set(data_h5ad.obs[self.random_type].tolist()))
                data_h5ad = data_h5ad[data_h5ad.obs[self.random_type].isin(self.type_list)]
            except FileNotFoundError as e:
                print(f"No such h5ad file found for [cyan]{dataset}")
                sys.exit(e)

            try:
                data_y = pd.DataFrame(data_h5ad.obs[self.random_type])
                data_y.reset_index(inplace=True, drop=True)
            except Exception as e:
                print(f"Celltype attribute not found for [cyan]{dataset}")
                sys.exit(e)

            if scipy.sparse.issparse(data_h5ad.X):
                data_x = pd.DataFrame(data_h5ad.X.todense())
            else:
                data_x = pd.DataFrame(data_h5ad.X)
            
            data_x = data_x.fillna(0) # fill na with 0    
            data_x.index = data_h5ad.obs_names
            data_x.columns = data_h5ad.var_names

            return data_x, data_y

        elif ".csv" in dataset:
            filename = os.path.join(self.data_path, dataset)

            try:
                data_x = pd.read_csv(filename, header=0, index_col=0)
            except FileNotFoundError as e:
                print(f"No such expression csv file found for [cyan]{dataset}")
                sys.exit(e)
        
            data_x = data_x.fillna(0) # fill na with 0    
            
            if metadata is not None:
                metadata_filename = os.path.join(self.data_path, metadata)
                try:
                    data_y = pd.read_csv(metadata_filename, header=0, index_col=0)
                except Exception as e:
                    print(f"Celltype attribute not found for [cyan]{dataset}")
                    sys.exit(e)
            else:
                print(f"Metadata file is not provided for [cyan]{dataset}")
                sys.exit(1)

            return data_x, data_y

    def load_real_data(self, dataset):
        
        if ".h5ad" in dataset:
            filename = os.path.join(self.data_path, dataset)

            try:
                data_h5ad = ad.read_h5ad(filename)
            except FileNotFoundError as e:
                print(f"No such h5ad file found for [cyan]{dataset}.")
                sys.exit(e)
            
            if scipy.sparse.issparse(data_h5ad.X):
                data_h5ad.X = pd.DataFrame(data_h5ad.X.todense()).fillna(0)
            else:
                data_h5ad.X = pd.DataFrame(data_h5ad.X).fillna(0)
        
            if self.normalize:
                data_h5ad.X = sample_normalize(data_h5ad.X, normalize_method=self.normalize)

            return data_h5ad

        elif ".csv" in dataset:
            filename = os.path.join(self.data_path, dataset)

            try:
                data_x = pd.read_csv(filename, header=0, index_col=0)
            except FileNotFoundError as e:
                print(f"No such target expression csv file found for [cyan]{dataset}.")
                sys.exit(e)
            
            data_x = data_x.fillna(0) # fill na with 0    
            
            data_h5ad = ad.AnnData(X=data_x)
            data_h5ad.var_names = data_x.columns

            if self.normalize:
                data_h5ad.X = sample_normalize(data_h5ad.X, normalize_method=self.normalize)

            return data_h5ad

    def mixup_fraction(self, celltype_num):

        fracs = np.random.rand(celltype_num)
        fracs_sum = np.sum(fracs)
        fracs = np.divide(fracs, fracs_sum)

        return fracs

    def mixup_cells(self, x, y, celltypes):

        available_celltypes = celltypes
        
        celltype_num = len(available_celltypes)

        # Create fractions for available celltypes
        fracs = self.mixup_fraction(celltype_num)

        samp_fracs = np.multiply(fracs, self.sample_size)
        samp_fracs = list(map(round, samp_fracs))
        
        # Make complete fracions
        fracs_complete = [0] * len(celltypes)

        for i, act in enumerate(available_celltypes):
            idx = celltypes.index(act)
            fracs_complete[idx] = fracs[i]

        artificial_samples = []

        for i in range(celltype_num):
            ct = available_celltypes[i]
            cells_sub = x.loc[np.array(y[self.random_type] == ct), :]
            cells_fraction = np.random.randint(0, cells_sub.shape[0], samp_fracs[i])
            cells_sub = cells_sub.iloc[cells_fraction, :]
            artificial_samples.append(cells_sub)

        df_samp = pd.concat(artificial_samples, axis=0)
        df_samp = df_samp.sum(axis=0)

        return df_samp, fracs_complete

    