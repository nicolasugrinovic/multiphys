import argparse
import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from scipy.spatial.transform import Rotation as sRot
import joblib
# from embodiedpose.models.humor.body_model.utils import smpl_to_openpose
# from embodiedpose.models.humor.utils.humor_mujoco import SMPL_2_OP, OP_14_to_OP_12
# from uhc.smpllib.smpl_parser import SMPL_Parser, SMPLH_BONE_ORDER_NAMES, SMPLH_Parser
import torch
from pathlib import Path
# from utils.misc import read_json
from utils.smpl import smpl_to_verts
from utils.misc import save_trimesh
from tqdm import tqdm

from utils.smpl_robot import from_qpos_to_verts_w_model
from utils.process_utils import load_humanoid
from pyquaternion import Quaternion as Q

from multiphys.process_data.preprocess.smpl_transforms import rot_and_correct_smplx_offset_full

from utils.net_utils import get_hostname
hostname = get_hostname()

torch.set_default_dtype(torch.float32)


np.set_printoptions(precision=4)
smplx_folder = '/home/nugrinovic/code/CVPR_2024/EmbodiedPose/data'

def get_paths():
    if "oriong" in hostname:
        SLA_ROOT = "/sailhome/nugrinov/code/CVPR_2024/slahmr_release/slahmr"
    else:
        SLA_ROOT = "/home/nugrinovic/code/CVPR_2024/slahmr_release/slahmr"
    VIDEOS_ROOT = f"{SLA_ROOT}/videos"
    return SLA_ROOT, VIDEOS_ROOT

def read_proc_data(PROC_ROOT, subj_name):
    sub_n = f"_{subj_name}" if subj_name != args.data_name else ""
    if args.data_name == 'chi3d' or args.data_name == 'hi4d':
        p1 = f"{PROC_ROOT}/{args.data_name}/thirdeye_clip_{args.data_name}{sub_n}_embodied_cam2w_p1_floor_phalp_initSla.pkl"
        p2 = f"{PROC_ROOT}/{args.data_name}/thirdeye_clip_{args.data_name}{sub_n}_embodied_cam2w_p2_floor_phalp_initSla.pkl"
    else:
        p1 = f"{PROC_ROOT}/{args.data_name}/thirdeye_clip_{args.data_name}{sub_n}_p1_phalpBox_all_slaInit_slaCam.pkl"
        p2 = f"{PROC_ROOT}/{args.data_name}/thirdeye_clip_{args.data_name}{sub_n}_p2_phalpBox_all_slaInit_slaCam.pkl"

    proc_data = [joblib.load(p1)]
    proc_data.append(joblib.load(p2))
    return proc_data


def build_simu_res_dict(seq_name, key_name, n, proc_this, all_results, mean_shape, models):
    debug = False
    
    kp_25 = proc_this['joints2d']
    qpos_pred = all_results[seq_name][n]['pred']
    betas = mean_shape[n].cpu().numpy()
    # this introduces an 180z degree rotation, need to compensate
    pose_aa, trans, _ = from_qpos_to_verts_w_model(qpos_pred, models[n], betas=betas[:10], ret_body_pose=True)
    Tz180 = Q(axis=[0, 0, 1], angle=np.pi).transformation_matrix.astype(np.float32)
    pose_aa_rot, trans_rot = rot_and_correct_smplx_offset_full(pose_aa, trans, betas[None, :10], Tz180[None])

    cam = proc_this['cam']
    B = len(pose_aa)
    start, end = 0, B

    pose_mat = sRot.from_rotvec(pose_aa_rot.reshape(B * 24, 3)).as_matrix().reshape(B, 24, 3, 3)
    pose_body = pose_mat[:, 1:22]
    root_orient = pose_mat[:, 0:1]

    if debug:
        op = f"inspect_out/process_simu"
        Path(op).parent.mkdir(parents=True, exist_ok=True)

        verts, faces = smpl_to_verts(pose_aa_rot[0, None], trans_rot[0, None], betas=betas[None, :10])
        save_trimesh(verts[0, 0], faces, f"{op}/pose_aa_rot_{n}.ply")
        verts, faces = smpl_to_verts(pose_aa[0, None], np.zeros_like(trans[0, None]), betas=betas[None, :10])
        save_trimesh(verts[0, 0], faces, f"{op}/pose_aa_root_rel_{n}.ply")

        pose_aa_pr = proc_this['pose_aa']
        trans_pr = proc_this['trans']
        betas_pr = proc_this['betas']
        verts, faces = smpl_to_verts(pose_aa_pr[0, None], trans_pr[0, None], betas=betas_pr[None, :10])
        save_trimesh(verts[0, 0], faces, f"{op}/proc/pose_proc_{n}.ply")

    out_dict= {
        "joints2d": kp_25.copy(),  # (B, 25, 3)

        "pose_aa": pose_aa_rot.reshape(-1, 72)[start:end],  # (B, 72)
        "root_orient": root_orient.squeeze(),  # (B, 3, 3)
        "pose_body": pose_body[start:end],  # (B, 21, 3, 3)
        "trans": trans_rot.squeeze()[start:end],  # (B, 3)
        'betas': betas.squeeze(),  # (16, )

        "pose_6d": np.zeros([B, 24, 6]),
        "trans_vel": np.zeros([B, 1, 3]),  # (B, 3)
        "root_orient_vel": np.zeros([B, 1, 3]),
        "joints": np.zeros([B, 22, 3]),  # (B, 66)
        "joints_vel": np.zeros([B, 22, 3]),
        "seq_name": key_name,
        "gender": "neutral",

        "points3d": None,
        "cam": cam
    }
    return out_dict


def process(args, debug=False):
    from metrics.prepare_slahmr_results import get_emb_paths
    from metrics.prepare import get_all_results

    smpl_robot, humanoid, cc_cfg = load_humanoid()

    count = 0
    if args.debug:
        print("*** DEBUG MODE ***")
        print("*** DEBUG MODE ***")
    data_name = args.data_name
    filter = args.filter_seq

    path_, SEQS, ROOT, RES_ROOT = get_emb_paths(args.data_name, args.exp_name)
    PROC_ROOT = f"{ROOT}/sample_data"

    for subj_name in SEQS:
        print(f"Doing {subj_name}...")

        path = f"{RES_ROOT}/{args.data_name}/{args.exp_name}/{subj_name}"
        path = Path(path)
        # read proc data
        proc_data = read_proc_data(PROC_ROOT, subj_name)
        # read result files
        print(f"Reading results from:\n {path}")
        all_results = get_all_results(path)

        for nn, seq_name in enumerate(tqdm(all_results)):
            results = all_results[seq_name]
            p_map = {0: 1, 1: 0}
            # Format data for saving
            for n in range(len(results)):

                seq_name = 's02_Grab_17'
                proc_this = proc_data[p_map[n]][seq_name]

                output_dir = f"{args.output_dir}/simu_output_01/{args.exp_name}"
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                subj_n = f"_{subj_name}" if data_name =='chi3d' or data_name=='expi' else ""
                data_n = data_name + "_slahmr" if data_name == 'chi3d' else data_name
                debug_n = "_debug" if args.debug else ""
                person_id_n = f"_p{n+1}"
                dyncam_n = "_dyncam" if args.dyncam else ""
                filt_n = f"_{args.filter_seq}" if filter else "_all"
                out_fpath = Path(output_dir) / f"{data_n}{subj_n}{debug_n}{person_id_n}{dyncam_n}{filt_n}_slaInit_slaCam.pkl"
                # read
                simu_data = joblib.load(out_fpath)
                simu_data_seq = simu_data[seq_name]


    print("Done")


def main(args):
    process(args)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default='chi3d', choices=['viw', 'chi3d', 'hi4d', 'expi', 'shorts'])
    parser.add_argument("--vid_name", type=str, default=None)
    # parser.add_argument("--subject", type=str, default='.')
    parser.add_argument("--output_dir", default='sample_data/videos_wild')
    parser.add_argument('--debug', type=int, choices=[0, 1], default=0)
    parser.add_argument("--joints_source", type=str, choices=['hybrik', 'phalp'], default='phalp')
    parser.add_argument("--bbox_source", type=str, choices=['hybrik', 'phalp'], default='phalp')
    parser.add_argument('--dyncam', type=int, choices=[0, 1], default=0)
    parser.add_argument("--filter", type=str, default=None)
    parser.add_argument("--filter_actions", type=int, choices=[0, 1], default=0)
    parser.add_argument("--exp_name", type=str, default='slahmr_override', choices=['slahmr_override',
                                                                              'slahmr_override_loop4'])
    args = parser.parse_args()

    main(args)