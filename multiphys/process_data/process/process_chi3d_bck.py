""" this file comes from embodiedpose/data_process/process_humor_split.py"""

import argparse
from enum import EnumMeta
import json
import os
import sys
import os
import sys
import pdb
import os.path as osp
import roma
sys.path.append(os.getcwd())
import glob
import joblib
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.getcwd())

from uhc.utils.image_utils import get_chunk_with_overlap
from scipy.ndimage import gaussian_filter1d

from utils.misc import read_pickle

import pandas as pd
from pathlib import Path
from utils.pyquaternion import Quaternion as Q
from math import pi
from utils.smpl import smpl_to_verts
from utils.misc import save_trimesh
from utils.misc import read_image_PIL, plot
from utils.net_utils import replace_slahmr_path
import trimesh

from constants import *
from utils.process_utils import load_humanoid
from utils.process_utils import smpl_2_entry

# smpl_robot, humanoid, cc_cfg = load_humanoid()
Tx = Q(axis=[0, 0, 1], angle=-pi / 2.).transformation_matrix.astype(np.float32)
Tx = torch.tensor(Tx[:3, :3])
np.random.seed(1)


def read_keypoints(keypoint_fn):
    '''
    Only reads body keypoint data of first person.
    '''
    with open(keypoint_fn) as keypoint_file:
        data = json.load(keypoint_file)

    if len(data['people']) == 0:
        print('WARNING: Found no keypoints in %s! Returning zeros!' % (keypoint_fn))
        return np.zeros((25, 3), dtype=np.float)

    person_data = data['people'][0]
    body_keypoints = np.array(person_data['pose_keypoints_2d'], dtype=np.float)
    body_keypoints = body_keypoints.reshape([-1, 3])

    return body_keypoints


def load_camera_params(scene_name):
    prox_path = './data/prox/qualitative'
    with open(f'{prox_path}/calibration/Color.json', 'r') as f:
        cameraInfo = json.load(f)
        K = np.array(cameraInfo['camera_mtx']).astype(np.float32)

    with open(f'{prox_path}/cam2world/{scene_name}.json', 'r') as f:
        camera_pose = np.array(json.load(f)).astype(np.float32)
        R = camera_pose[:3, :3]
        tr = camera_pose[:3, 3]
        R = R.T
        tr = -np.matmul(R, tr)

    with open(f'{prox_path}/alignment/{scene_name}.npz', 'rb') as f:
        aRt = np.load(f)
        aR = aRt['R']
        atr = aRt['t']
        aR = aR.T
        atr = -np.matmul(aR, atr)
    full_R = R.dot(aR)
    full_t = R.dot(atr) + tr

    cam_params = {"K": K, "R": R, "tr": tr, "aR": aR, "atr": atr, "full_R": full_R, "full_t": full_t}
    return cam_params



def print_keys(motion_files):
    data4stats = read_pickle(motion_files[0])
    for k, v in data4stats.items():
        print(f"'{k}': {v['trans'].shape[0]},")
    # camera_pose = data["cam_50591643"]

def save_file(args, out_path, data_res):
    debug_name = "_debug" if args.debug else ""
    floor_name = "_floor" if args.to_floor else ""
    smooth_name = "_smooth2d" if args.smooth_2d else ""
    masked_name = "_masked" if args.mask_2d else ""
    jts_name = f"_{args.joints_source}"
    person_id_name = f"_p{args.person_id}"
    subset_name = "_subset" if args.subset else ""
    output_file_name = osp.join(out_path, f'thirdeye_clip_{args.fname}{person_id_name}{debug_name}'
                                          f'{floor_name}{smooth_name}'
                                          f'{masked_name}{jts_name}{subset_name}.pkl')
    print(output_file_name, len(data_res))
    Path(output_file_name).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(data_res, open(output_file_name, "wb"))

def trans_to_floor(smpl_dict, debug=False):
    pose = smpl_dict['pose']
    trans = smpl_dict['trans']
    betas = smpl_dict['shape']
    betas = torch.from_numpy(betas).float()  # .cuda()
    verts, faces = smpl_to_verts(pose, trans, betas=betas[0, None, :10])
    mesh = trimesh.Trimesh(vertices=verts[0, 0].detach().cpu().numpy(), faces=faces)
    bbox = mesh.bounding_box.bounds
    min_xyz = bbox[0]
    trans_floor_z = trans[:, 2] - min_xyz[None, 2]
    new_trans = np.concatenate([trans[:, :2], trans_floor_z[:, None]], 1)

    if debug:
        verts, faces = smpl_to_verts(pose[0, None], new_trans[0, None])
        save_trimesh(verts[0, 0], faces, f"inspect_out/chi3d/meshes/process/floor{person_name}.ply")

    return new_trans

def smooth_2dkpts(seq_length, smpl_dict):
    chunk_bounds, selects = get_chunk_with_overlap(seq_length, 60, 10, True)
    for curr_seq in chunk_bounds:
        window = 30
        if curr_seq[0] != 0:
            filter_in = smpl_dict['joints2d'][(curr_seq[0] - window):(curr_seq[0] + window)]
            filter_in = filter_in.reshape(-1, 25 * 3)
            filter_out = gaussian_filter1d(filter_in, 4, axis=0)
            if 0:
                plt.plot(filter_in[:, 0], 'k', label='original data')
                plt.plot(filter_out[:, 0], '--', label='filtered, sigma=3')
                plt.legend()
                plt.grid()
                plt.show()
            filter_out = filter_out.reshape(-1, 25, 3)
            smpl_dict['joints2d'][(curr_seq[0] - window):(curr_seq[0] + window)] = filter_out

def mask_2d(len_seq, joints2d, smpl_dict):
    wind_size = 10
    max_start = len_seq - wind_size
    n_candidates = 20
    starts = []
    for i in range(n_candidates):
        start = np.random.choice(range(0, max_start))
        starts.append(start)
    remove = []
    for i in range(len(starts)):
        rem_this = [False for _ in range(i)]
        for j in range(i + 1, len(starts)):
            diff = np.abs(starts[i] - starts[j])
            if diff < wind_size + 10:
                rem_this.append(True)
            else:
                rem_this.append(False)
        remove.append(rem_this)
    remove = np.array(remove)
    remove_ = np.any(remove, axis=1)
    # remove the ones that are too close
    starts = np.array(starts)[~remove_]
    starts = np.sort(starts)
    diffs = np.diff(starts)
    mask = np.zeros_like(joints2d[:, 0, 0]).astype(bool)
    for sta in starts:
        mask[sta:sta + wind_size] = np.ones([wind_size]).astype(bool)
    mask = np.logical_not(mask).astype(np.float32)
    joints2d_masked = joints2d * mask[:, None, None]
    smpl_dict['joints2d'] = joints2d_masked


def main(args, debug=False):
    SLA_ROOT = "/sailhome/nugrinov/code/CVPR_2024/slahmr_release/slahmr/videos"

    data_res = {}
    count = 0
    limit = args.limit
    out_path = './sample_data/chi3d/'

    # filter for debugging
    # SEQS = [c for c in SEQS if 'Hug' in c]
    person_name = f"_p{args.person_id}"
    dataset_path = f'./data/chi3d/{args.fname}{person_name}.pkl'
    dataset_data = read_pickle(dataset_path)

    sub_name = args.fname.split('_')[1]
    SEQS = list(SEQ_LENGTHS[sub_name])
    pbar = tqdm(SEQS)
    for idx, seq in enumerate(pbar):
        pbar.set_description(seq)
        if limit == -1:
            pass
        elif idx > limit:
            print(f"*** Warning: Breaking after {limit} seq ***")
            continue
        if not os.path.exists(dataset_path):
            print("Empty motion files for seq:", seq, idx)
            continue
            
        # data --> keys: (['s02_Grab_1', 's02_Grab_10', 's02_Grab_11'])
        # motion --> keys: ['trans', 'root_orient', 'pose_body', 'betas', 'joints2d', 'cam2world']
        motion = dataset_data[seq]
        camera_pose = motion["cam2world"]
        seq_length = SEQ_LENGTHS[sub_name][seq]
        # CHI3D has no gender data
        gender = 'neutral'
        smpl_dict = {'pose': np.zeros((seq_length, 72)), 'trans': np.zeros((seq_length, 3)),
                     'shape': np.zeros((seq_length, 16)), 'joints2d': np.zeros((seq_length, 25, 3)),
                     'gender': gender}
        if args.dataset == 'proxd':
            smpl_dict['points3d'] = np.zeros((seq_length, 4096, 3))

        # motion --> contains ['betas', 'trans', 'root_orient', 'pose_body', 'floor_plane', 'contacts']
        len_seq = smpl_dict['joints2d'].shape[0]
        #####
        # from motion data, emb uses --> root_orient, pose_body, trans, betas
        # NOTE: pose is 72 dim, but root_orient+pose_body is 66 dim, so the last 6 dims are not used and set to 0
        root_orient = motion['root_orient']
        smpl_dict['pose'][:, :3] = root_orient
        smpl_dict['pose'][:, 3:66] = motion['pose_body']
        smpl_dict['trans'][:] = motion['trans']
        smpl_dict['shape'][:][:, :10] = motion['betas']  # Betas!!
        smpl_dict['joints2d'] = motion["joints2d"]
        joints2d = smpl_dict['joints2d']
        seq_name = seq[4:]

        # load SLAHMR data

        
        ### Load 2D kpts
        # The detections here are multiperson, so we descard DEKR
        if args.joints_source == 'smpl_gt':
            # GT kpts do not correspond to EmbPose joints definition, thus result in a lot of jitter
            smpl_dict['joints2d'] = motion["joints2d"]
        elif args.joints_source == 'phalp':
            # PHALP estimates
            # phalp_path = f"/home/nugrinovic/code/CVPR_2024/EmbodiedPose/sample_data/chi3d_phalp_data/" \
            #              f"train/{sub_name}/slahmr/results/phalp_out/results/{seq_name}.pkl"
            phalp_path = f"{SLA_ROOT}/chi3d/train/{sub_name}/slahmr/phalp_out/results/{seq_name}.pkl"
            try:
                data = joblib.load(phalp_path)
            except:
                print(f"File {phalp_path} does not exist")
                continue
            img_names = sorted(data.keys())

            all_joints = []
            is_complete = []
            for im_name in img_names:
                phalp_data = data[im_name]
                vitpose = phalp_data['vitpose']  # list of 2 with shape (25, 3)
                # vitpose only has joints from 0 to 18, so 19 jts and does not have 1 (chest)
                # nor 8 (root), it does contain the nose at 0.
                vitpose_arr = np.zeros([2, 25, 3])
                complete = False
                len_vit = len(vitpose)
                if len_vit > 2:
                    print("More than 2 people in the frame")
                for ip, pose_list in enumerate(vitpose[:2]):
                    vitpose_arr[ip] = vitpose[ip]
                    if ip == 1:
                        complete = True
                # vitpose_arr = np.array(vitpose)
                is_complete.append(complete)
                all_joints.append(vitpose_arr)
            op_kpts = np.stack(all_joints, 0)
            # am I using the conf values here?
            # print(op_kpts[0, 0]) # yes! confs are taken directly from the vitpose output
            op_kpts[:, :, 1] = (op_kpts[:, :, 2] + op_kpts[:, :, 5]) / 2
            op_kpts[:, :, 8] = (op_kpts[:, :, 9] + op_kpts[:, :, 12]) / 2

            # update this according to the person id, is this important though? yes to match the smpl gt
            try:
                complete_idx = np.where(np.array(is_complete))[0][0]
            except:
                print(f"Detections contain only one person, using the first frame")
                complete_idx = 0

            kpts_gt_first = smpl_dict['joints2d'][complete_idx]
            kpts_vit = op_kpts[complete_idx]
            # l2 dist between GT and vitpose, to match the person
            l2_dist = np.linalg.norm(kpts_gt_first[None] - kpts_vit, axis=-1).mean(1)
            min_id = np.argmin(l2_dist)
            kpts_this_person = op_kpts[:seq_length, min_id]


            # vitpose[0, 19]
            if debug:
                from utils.misc import plot_joints_cv2
                idx = complete_idx
                res_img_path = Path(
                    f"/home/nugrinovic/code/NEURIPS_2023/slahmr/videos/chi3d/train/s02/images/{seq_name.replace('_', ' ')}")
                img_files = sorted(res_img_path.glob("*.jpg"))
                img = read_image_PIL(img_files[idx])
                # plot(img)
                img_w_jts = plot_joints_cv2(img, kpts_this_person[idx, None], show=True, with_text=True, sc=2)
                j2d = smpl_dict['joints2d']
                img_w_jts = plot_joints_cv2(img, j2d[idx, None], show=True, with_text=True, sc=2)

            smpl_dict['joints2d'] = kpts_this_person  # (219, 25, 3)

        else:
            print(f"Unknown joints source {args.joints_source}")
            raise NotImplementedError

        if args.mask_2d:
            mask_2d(len_seq, joints2d, smpl_dict)

        if args.smooth_2d:
            smooth_2dkpts(seq_length, smpl_dict)

        if args.to_floor:
            new_trans = trans_to_floor(smpl_dict)
            smpl_dict['trans'] = new_trans

        if debug:
            pose = smpl_dict['pose']
            trans = smpl_dict['trans']
            verts, faces = smpl_to_verts(pose, trans)
            save_trimesh(verts[0, 0], faces, "inspect_out/chi3d/meshes/smpl_gt.ply")

        entry = smpl_2_entry(seq, smpl_dict, camera_pose)
        
        if args.debug:
            count += 1
            excluded = ["betas", "cam", "points3d", "gender", "seq_name"]
            for k, v in entry.items():
                if k in excluded:
                    continue
                entry[k] = v[:31]
            if count > 3:
                break

        data_res[seq] = entry

    print(data_res.keys())
    save_file(args, out_path, data_res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=["prox", "proxd", "video"], default="proxd")
    parser.add_argument('--fname', type=str)
    parser.add_argument('--sub_name', type=str)
    parser.add_argument('--debug', type=int, choices=[0, 1], default=0)
    parser.add_argument('--to_floor', type=int, choices=[0, 1], default=0)
    parser.add_argument('--smooth_2d', type=int, choices=[0, 1], default=0)
    parser.add_argument('--mask_2d', type=int, choices=[0, 1], default=0)
    parser.add_argument('--person_id', type=int, choices=[1, 2], default=1)
    parser.add_argument('--joints_source', type=str, choices=["dekr", "smpl_gt", "phalp"], default="smpl_gt")
    parser.add_argument('--limit', type=int, default=-1)
    parser.add_argument('--subset', type=int, choices=[0, 1], default=0)
    args = parser.parse_args()
    
    main(args)

    
