import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import joblib
from utils.net_utils import replace_slahmr_path
from tqdm import tqdm

np.set_printoptions(precision=4)
smplx_folder = '/home/nugrinovic/code/CVPR_2024/EmbodiedPose/data'

def read_PHALP_data(VIDEOS_ROOT, data_name, subject, seq_name):
    # use PHALP bbox here, so that it can work with multiple people
    if data_name == 'chi3d':
        # phalp_path = f"{VIDEOS_ROOT}/{data_name}/train/{subject}/slahmr/phalp_out/results/{seq_name}.pkl"
        phalp_path = f"{VIDEOS_ROOT}/{data_name}/train/{subject}/phalp_light/{seq_name}.pkl"
    else:
        # phalp_path = f"/home/nugrinovic/code/NEURIPS_2023/slahmr/videos/{data_name}/slahmr/phalp_out/results/{seq_name}.pkl"
        phalp_path = f"/home/nugrinovic/code/NEURIPS_2023/slahmr/videos/{data_name}/{subject}/phalp_light/{seq_name}.pkl"
        phalp_path = replace_slahmr_path(phalp_path)
    try:
        phalp_data = joblib.load(phalp_path)
    except:
        tqdm.write(f"skipping {seq_name}, phalp LIGHT results missing. Generate in slahmr!")
        return None
    return phalp_data


