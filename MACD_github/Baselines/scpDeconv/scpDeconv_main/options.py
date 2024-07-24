from collections import defaultdict

option_list = defaultdict(list)

def get_option_list(dataset):
	
	if dataset == 'human_breast_atlas_PP':
		# Input parameters
		option_list['data_dir']='./data/human_breast_atlas_PP/'
		option_list['ref_dataset_name'] = 'Human_Breast_Atlas_scProteome_normed_aligned_individual1.h5ad'
		option_list['ref_metadata_name'] = None
		option_list['target_dataset_name'] = 'Human_Breast_Atlas_scProteome_normed_aligned_individual3.h5ad'
		option_list['target_metadata_name'] = None
		option_list['random_type']="cell_type"
		option_list['type_list']=None
		# Training parameters
		option_list['ref_sample_num'] = 4000
		option_list['sample_size'] = 50
		option_list['HVP_num'] = 0
		option_list['target_type'] = "simulated"
		option_list['target_sample_num'] = 1000	
		option_list['batch_size'] = 50
		option_list['epochs'] = 30
		option_list['learning_rate'] = 0.0001
		# Output parameters
		option_list['SaveResultsDir'] = "./Result/human_breast_atlas_PP/"

	elif dataset == 'murine_cellline':
		# Input parameters
		option_list['data_dir']='./data/murine_cellline/'
		option_list['ref_dataset_name'] = 'murine_N2_SCP_exp.csv'
		option_list['ref_metadata_name'] = 'murine_N2_SCP_meta.csv'
		option_list['target_dataset_name'] = 'murine_nanoPOTS_SCP_exp.csv'
		option_list['target_metadata_name'] = 'murine_nanoPOTS_SCP_meta.csv'
		option_list['random_type']="CellType"
		option_list['type_list']=['C10','SVEC','RAW']
		# Training parameters
		option_list['ref_sample_num'] = 4000
		option_list['sample_size'] = 15
		option_list['HVP_num'] = 500
		option_list['target_type'] = "simulated"
		option_list['target_sample_num'] = 1000	
		option_list['batch_size'] = 50
		option_list['epochs'] = 30
		option_list['learning_rate'] = 0.0001
		# Output parameters
		option_list['SaveResultsDir'] = "./Result/murine_cellline/"

	elif dataset == 'human_cellline':
		# Input parameters
		option_list['data_dir']='./data/human_cellline/'
		option_list['ref_dataset_name'] = 'pSCoPE_Huffman_PDAC+pSCoPE_Leduc+SCoPE2_Leduc_integrated_SCP.h5ad'
		option_list['ref_metadata_name'] = None
		option_list['target_dataset_name'] = 'T-SCP+plexDIA_integrated_SCP.h5ad'
		option_list['target_metadata_name'] = None
		option_list['random_type']="cell_type"
		option_list['type_list']=None
		# Training parameters
		option_list['ref_sample_num'] = 4000
		option_list['sample_size'] = 50
		option_list['HVP_num'] = 500
		option_list['target_type'] = "simulated"
		option_list['target_sample_num'] = 1000	
		option_list['batch_size'] = 50
		option_list['epochs'] = 30
		option_list['learning_rate'] = 0.0001
		# Output parameters
		option_list['SaveResultsDir'] = "./Result/human_cellline/"

	elif dataset == 'simulated_data':
		# Input parameters
		option_list['data_dir']='./data/human_cellline/'
		option_list['ref_dataset_name'] = 'pSCoPE_Huffman_PDAC+pSCoPE_Leduc+SCoPE2_Leduc_integrated_SCP.h5ad'
		option_list['ref_metadata_name'] = None
		option_list['target_dataset_name'] = 'T-SCP+plexDIA_integrated_SCP.h5ad'
		option_list['target_metadata_name'] = None
		option_list['random_type']="cell_type"
		option_list['type_list']=None
		# Training parameters
		option_list['ref_sample_num'] = 4000
		option_list['sample_size'] = 50
		option_list['HVP_num'] = 500
		option_list['target_type'] = "simulated"
		option_list['target_sample_num'] = 1000
		option_list['batch_size'] = 50
		option_list['epochs'] = 30
		option_list['learning_rate'] = 0.0001
		# Output parameters
		option_list['SaveResultsDir'] = "./Result/human_cellline/"

	return option_list
