
# need to export these for mujoco
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia:/home/nugrinovic/.mujoco/mujoco210/bin;
export MUJOCO_GL='egl';

# process videos for visualization
python multiphys/process_data/process_videos.py

# generate sequences
# expi sequences
python run.py --cfg tcn_voxel_4_5_chi3d_multi_hum --data sample_data/expi/expi_acro1_p1_phalpBox_all_slaInit_slaCam.pkl --data_name expi --name slahmr_override_loop2 --loops_uhc 2 --filter acro1_around-the-back1_cam20
# ground-penet: acro1_frog-turn1_cam20
python run.py --cfg tcn_voxel_4_5_chi3d_multi_hum --data sample_data/expi/expi_acro1_p1_phalpBox_all_slaInit_slaCam.pkl --data_name expi --name slahmr_override_loop2 --loops_uhc 2 --filter acro1_frog-turn1_cam20
# chi3d sequences
# spatial placement: s03_Hug_7
python run.py --cfg tcn_voxel_4_5_chi3d_multi_hum --data sample_data/chi3d_slahmr/chi3d_slahmr_s03_p1_phalpBox_all_slaInit_slaCam.pkl --data_name chi3d --name slahmr_override --loops_uhc 2 --filter s03_Hug_7
# spatial placement: s02_Hug_5
python run.py --cfg tcn_voxel_4_5_chi3d_multi_hum --data sample_data/chi3d_slahmr/chi3d_slahmr_s02_p1_phalpBox_all_slaInit_slaCam.pkl --data_name chi3d --name slahmr_override_loop2 --loops_uhc 2 --filter s02_Hug_5
# hi4d sequences
python run.py --cfg tcn_voxel_4_5_chi3d_multi_hum --data sample_data/hi4d_slahmr/hi4d_p1_phalpBox_all_slaInit_slaCam.pkl --data_name hi4d --name slahmr_override_loop2 --loops_uhc 2 --filter _pair02_2_pose02
python run.py --cfg tcn_voxel_4_5_chi3d_multi_hum --data sample_data/hi4d_slahmr/hi4d_p1_phalpBox_all_slaInit_slaCam.pkl --data_name hi4d --name slahmr_override_loop2 --loops_uhc 2 --filter _pair21_2_pose21
python run.py --cfg tcn_voxel_4_5_chi3d_multi_hum --data sample_data/hi4d_slahmr/hi4d_p1_phalpBox_all_slaInit_slaCam.pkl --data_name hi4d --name slahmr_override_loop2 --loops_uhc 2 --filter pair12_talk12
