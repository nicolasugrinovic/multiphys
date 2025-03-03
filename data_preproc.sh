# First make sure that you already generated the SLAHMR estimates,
# then you should generate the scene_dict.pkl files by running the following command (NOTE that you need to
# add two scripts to your SLAHMR repo to do this):
# bash scripts/run_opt_world_{dataset_name}.sh args
# for each dataset
# The execute the following:

############################## CHI3D #####################
# PROCESS DATASET FILES
python multiphys/data_process/chi3d_for_emb.py
# PROCESS PHALP FILES - CHI3D
python multiphys/data_process/process_phalp_all.py --output sample_data/chi3d_slahmr --data_name chi3d


########################## HI4D ##########################
# PROCESS DATASET FILES
python multiphys/data_process/hi4d_for_emb.py
# PROCESS PHALP FILES - HI4D
python multiphys/data_process/process_phalp_all.py --output sample_data/hi4d_slahmr --data_name hi4d


############################## EXPI ########################
# PROCESS DATASET FILES
python multiphys/data_process/expi_for_emb.py
# PROCESS PHALP DATA - Expi
python multiphys/data_process/process_phalp_all.py --output sample_data/expi --data_name expi