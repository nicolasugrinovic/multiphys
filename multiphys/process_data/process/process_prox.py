""" this file comes from embodiedpose/data_process/process_humor_split.py
The goal of this script is to see if replicating the preprocessing the method
works well.
"""

import argparse
from enum import EnumMeta
import json
import os
import sys
import os
import sys
import pdb
import os.path as osp

sys.path.append(os.getcwd())

import glob
import joblib
import numpy as np
from scipy.spatial.transform import Rotation as sRot
import torch
from tqdm import tqdm
from mujoco_py import load_model_from_path

sys.path.append(os.getcwd())

from uhc.utils.torch_ext import dict_to_numpy
from uhc.utils.transform_utils import smooth_smpl_quat_window
from uhc.utils.image_utils import get_chunk_with_overlap
from scipy.ndimage import gaussian_filter1d
from uhc.smpllib.smpl_robot import Robot
from uhc.smpllib.torch_smpl_humanoid import Humanoid
import mujoco_py
from uhc.utils.transform_utils import (convert_aa_to_orth6d, rotation_matrix_to_angle_axis)
from uhc.smpllib.smpl_parser import SMPL_Parser
from uhc.utils.torch_geometry_transforms import (angle_axis_to_rotation_matrix)

from uhc.smpllib.smpl_mujoco import smpl_to_qpose_torch
from embodiedpose.models.humor.utils.humor_mujoco import reorder_joints_to_humor, MUJOCO_2_SMPL
from embodiedpose.models.humor.utils.velocities import estimate_velocities
from uhc.utils.config_utils.copycat_config import Config as CC_Config

from utils.misc import read_pickle, write_pickle, read_json, write_json
import trimesh

from utils.smpl import smpl_to_verts
from utils.misc import save_trimesh
from pathlib import Path

from pyquaternion import Quaternion as Q
from math import pi


np.random.seed(1)
LEFT_RIGHT_IDX = [
    0,
    2,
    1,
    3,
    5,
    4,
    6,
    8,
    7,
    9,
    11,
    10,
    12,
    14,
    13,
    15,
    17,
    16,
    19,
    18,
    21,
    20,
    23,
    22,
]
FEMALE_SUBJ_IDS = [162, 3452, 159, 3403]

SEQ_LENGTHS = {
    "BasementSittingBooth_00142_01": 1581,
    "BasementSittingBooth_00145_01": 2180,
    "BasementSittingBooth_03403_01": 2399,
    "BasementSittingBooth_03452_01": 2456,
    "MPH11_00034_01": 2174,
    "MPH11_00150_01": 2187,
    "MPH11_00151_01": 1458,
    "MPH11_00153_01": 2027,
    "MPH11_00157_01": 2000,
    "MPH11_00169_01": 1855,
    "MPH11_03515_01": 1914,
    "MPH112_00034_01": 1811,
    "MPH112_00150_01": 1752,
    "MPH112_00151_01": 1290,
    "MPH112_00157_01": 1241,
    "MPH112_00169_01": 1765,
    "MPH112_03515_01": 1570,
    "MPH16_00157_01": 1785,
    "MPH16_03301_01": 1518,
    "MPH1Library_00034_01": 2203,
    "MPH1Library_00145_01": 3483,
    "MPH1Library_03301_01": 1509,
    "MPH8_00034_01": 2949,
    "MPH8_00168_01": 3071,
    "MPH8_03301_01": 2126,
    "N0SittingBooth_00162_01": 1484,
    "N0SittingBooth_00169_01": 1096,
    "N0SittingBooth_00169_02": 1243,
    "N0SittingBooth_03301_01": 1088,
    "N0SittingBooth_03403_01": 1353,
    "N0Sofa_00034_01": 3038,
    "N0Sofa_00034_02": 1446,
    "N0Sofa_00141_01": 2266,
    "N0Sofa_00145_01": 2104,
    "N0Sofa_03403_01": 1603,
    "N3Library_00157_01": 967,
    "N3Library_00157_02": 714,
    "N3Library_03301_01": 827,
    "N3Library_03301_02": 652,
    "N3Library_03375_01": 1100,
    "N3Library_03375_02": 464,
    "N3Library_03403_01": 670,
    "N3Library_03403_02": 984,
    "N3Office_00034_01": 2152,
    "N3Office_00139_01": 1337,
    "N3Office_00139_02": 2254,
    "N3Office_00150_01": 2599,
    "N3Office_00153_01": 3060,
    "N3Office_00159_01": 2037,
    "N3Office_03301_01": 2044,
    "N3OpenArea_00157_01": 996,
    "N3OpenArea_00157_02": 1325,
    "N3OpenArea_00158_01": 1311,
    "N3OpenArea_00158_02": 1915,
    "N3OpenArea_03301_01": 1056,
    # "N3OpenArea_03403_01": 872,
    "Werkraum_03301_01": 904,
    "Werkraum_03403_01": 901,
    "Werkraum_03516_01": 1991,
    "Werkraum_03516_02": 1531,

}
SEQS = list(SEQ_LENGTHS.keys())

# TRIM_EDGES = 90
TRIM_EDGES = 0
IMG_WIDTH = 1920
OP_FLIP_MAP = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10, 11, 16, 15, 18, 17, 22, 23, 24, 19, 20, 21]


def left_to_rigth_euler(pose_euler):
    pose_euler[:, :, 0] = pose_euler[:, :, 0] * -1
    pose_euler[:, :, 2] = pose_euler[:, :, 2] * -1
    pose_euler = pose_euler[:, LEFT_RIGHT_IDX, :]
    return pose_euler


def transform_smpl(smpl_dict, R, t, seq_length, offset=[0, 0, 0]):
    offset = torch.tensor(offset).float()
    offset_rep = offset.repeat(seq_length, 1)[:, :, None]

    R_rep = R.repeat(seq_length, 1, 1)
    pose_orth = angle_axis_to_rotation_matrix(torch.from_numpy(smpl_dict['pose'].astype(np.float32)[:, :3]).reshape(-1, 3))
    pose_orth = pose_orth[:, :3, :3]
    pose_orth = torch.bmm(R_rep, pose_orth)
    trans = torch.from_numpy(smpl_dict['trans'].astype(np.float32)).reshape(-1, 3, 1)
    trans = torch.bmm(R_rep, (trans + offset_rep)) - offset_rep
    trans = trans[:, :, 0] + t[None]

    pose = np.array(rotation_matrix_to_angle_axis(pose_orth).reshape(seq_length, 3))
    trans = np.array(trans)
    return pose, trans


def smpl_2_entry(
        args,
    seq_name,
    smpl_dict,
):
    pose_aa = smpl_dict["pose"]
    trans = smpl_dict["trans"]
    seq_len = pose_aa.shape[0]
    shape = smpl_dict["shape"] if "shape" in smpl_dict else np.zeros([seq_len, 10])
    # mean_shape = shape.mean(axis=0)
    mean_shape = shape[0:10].mean(axis=0)
    # import ipdb; ipdb.set_trace()
    gender = smpl_dict["gender"] if "gender" in smpl_dict else "neutral"
    joints2d = smpl_dict["joints2d"] if "joints2d" in smpl_dict else None
    points3d = smpl_dict["points3d"] if "points3d" in smpl_dict else None
    th_betas = smpl_dict['shape']

    if 0:
        from utils.misc import plot_joints_cv2
        black = np.zeros([1080, 1920, 3], dtype=np.uint8)
        j2d = joints2d[0]
        plot_joints_cv2(black, joints2d[0, None], show=True, with_text=True, sc=3)

    seq_length = pose_aa.shape[0]
    if seq_length < 10:
        return None
    pose_aa = torch.from_numpy(pose_aa).float()
    pose_seq_6d = convert_aa_to_orth6d(pose_aa).reshape(-1, 144)
    smpl_robot.load_from_skeleton(torch.from_numpy(mean_shape[None,]), gender=[0], objs_info=None)
    model = mujoco_py.load_model_from_xml(smpl_robot.export_xml_string().decode("utf-8"))
    humanoid = Humanoid(model=model)
    qpos = smpl_to_qpose_torch(pose_aa, model, trans=torch.from_numpy(trans), count_offset=True)
    fk_result = humanoid.qpos_fk(qpos, to_numpy=False)

    root_orient = pose_aa[:, :3]
    joints_humor = reorder_joints_to_humor(fk_result['wbpos'].clone(), model, cc_cfg.robot_cfg.get("model", "smpl"))[:, :66]

    trans_vel, joints_humor_vel, root_orient_vel = estimate_velocities(torch.from_numpy(trans[None]), root_orient[None], joints_humor[None], 30)
    trans_vel = trans_vel[0]
    joints_humor_vel = joints_humor_vel[0]
    root_orient_vel = root_orient_vel[0]

    root_orient_mat = angle_axis_to_rotation_matrix(root_orient)[:, :3, :3]
    pose_body = pose_aa[:, 3:].reshape(-1, 23, 3)[:, :21]
    pose_body = angle_axis_to_rotation_matrix(pose_body.reshape(-1, 3))
    pose_body = pose_body.reshape(-1, 21, 4, 4)[:, :, :3, :3]

    camera_params = load_camera_params(seq_name[:-9])

    if args.scene:
        scene_name = seq_name[:-9]
    else:
        scene_name = "no_scene"

    entry = {
        # "expert": fk_result,
        "pose_aa": pose_aa,
        "pose_6d": pose_seq_6d,
        "pose_body": pose_body,
        "trans": trans,
        "trans_vel": trans_vel,
        "root_orient": root_orient_mat,
        "root_orient_vel": root_orient_vel,
        "joints": joints_humor,
        "joints_vel": joints_humor_vel,
        "betas": mean_shape,
        "seq_name": seq_name,
        "gender": gender,
        "joints2d": joints2d,
        "points3d": points3d,
        "cam": {
            "full_R": camera_params['full_R'],
            "full_t": camera_params['full_t'],
            "K": camera_params['K'],
            "img_w": 1980,
            "img_h": 1080,
            "scene_name": scene_name
        }
    }
    return dict_to_numpy(entry)


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
    # prox_path = "/hdd/zen/data/video_pose/prox/qualitative"
    prox_path = '/home/nugrinovic/code/CVPR_2024/EmbodiedPose/data/prox/qualitative/'

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

    # very important - This cameras are used in the simulation
    if args.ignore_align:
        full_R = R
        full_t = tr
        aR = np.zeros_like(aR)
        atr = np.zeros_like(atr)
    else:
        full_R = R.dot(aR)
        full_t = R.dot(atr) + tr

    cam_params = {"K": K, "R": R, "tr": tr, "aR": aR, "atr": atr, "full_R": full_R, "full_t": full_t}
    return cam_params


def load_humanoid():
    # cc_cfg = CC_Config(cfg_id="copycat_e_1", base_dir="../Copycat")
    cc_cfg = CC_Config(cfg_id="copycat_eval_1", base_dir="/")

    smpl_robot = Robot(
        cc_cfg.robot_cfg,
        data_dir=osp.join(cc_cfg.base_dir, "data/smpl"),
        masterfoot=cc_cfg.masterfoot,
    )
    model = mujoco_py.load_model_from_xml(smpl_robot.export_xml_string().decode("utf-8"))
    humanoid = Humanoid(model=model)
    return smpl_robot, humanoid, cc_cfg

def get_prox_pose_data():
    from utils.misc import read_pickle

    path = "data/prox/PROXD/N0Sofa_00141_01/results/s001_frame_00001__00.00.00.028/000.pkl"
    # data --> keys: (['camera_rotation', 'camera_translation', 'betas', 'global_orient', 'transl', 'left_hand_pose',
    # 'right_hand_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'expression', 'pose_embedding', 'body_pose'])
    data = read_pickle(path)
    transl = data["transl"]
    global_orient = data["global_orient"]
    body_pose = data["body_pose"]
    betas = data["betas"]
    camera_rotation = data["camera_rotation"]
    camera_translation = data["camera_translation"]
    # NOTE: Humor dict keys used are trans, pose_body, root_orient
    humor = {
        "trans": transl,
        "pose_body": body_pose,
        "root_orient": global_orient,
    }
    return humor

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=["prox", "proxd"], default="proxd")
    parser.add_argument('--flip_joints', type=int, choices=[0, 1], default=0)
    parser.add_argument('--override_2d', type=int, choices=[0, 1], default=0)
    # parser.add_argument('--use_dekr', type=int, choices=[0, 1], default=1)
    parser.add_argument('--joints_source', type=str, choices=["dekr", "smpl_gt"], default=None)
    parser.add_argument('--ignore_align', type=int, choices=[0, 1], default=0)
    parser.add_argument('--to_floor', type=int, choices=[0, 1], default=0)
    parser.add_argument('--noisy_rotz', type=int, choices=[0, 1], default=0)
    parser.add_argument('--noisy_rotx', type=int, choices=[0, 1], default=0)
    parser.add_argument('--set_conf_ones', type=int, choices=[0, 1], default=0)
    parser.add_argument('--noisy_2d', type=float, default=None)
    parser.add_argument('--debug_frames', type=int, default=31)
    parser.add_argument('--debug', type=int, choices=[0, 1], default=0)
    parser.add_argument('--scene', type=int, choices=[0, 1], default=0)
    # parser.add_argument('--seq_name', type=str)
    args = parser.parse_args()

    # pi=180
    z_angle = -pi
    Tz = Q(axis=[0, 0, 1], angle=z_angle).transformation_matrix.astype(np.float32)
    x_angle = -pi / 2.
    x_angle = 0
    Tx = Q(axis=[1, 0, 0], angle=x_angle).transformation_matrix.astype(np.float32)
    # T = torch.from_numpy(T[None, :3, :3]).cuda()

    smpl_robot, humanoid, cc_cfg = load_humanoid()
    data_res = {}
    seq_length = -1
    video_annot = {}
    seq_counter = 0
    count = 0
    single_seq_length = 60
    fix_feet = True

    prox_base = '/home/nugrinovic/code/CVPR_2024/EmbodiedPose/data/prox/qualitative/'

    pbar = tqdm(SEQS)
    for idx, seq in enumerate(pbar):
        pbar.set_description(seq)
        seq_name = seq

        result_path = f"/home/nugrinovic/code/CVPR_2024/EmbodiedPose/data/prox_embodied/{seq_name}/prox_humor_fmt.pkl"
        motion_files = sorted(glob.glob(result_path))

        if len(motion_files) == 0:
            print("Empty motion files for seq:", seq, idx)
            continue

        with open(osp.join(prox_base, f'cam2world/{seq[:-9]}.json'), 'r') as f:
            camera_pose = np.array(json.load(f)).astype(np.float32)
            R = torch.from_numpy(camera_pose[:3, :3])
            t = torch.from_numpy(camera_pose[:3, 3])

        if not args.ignore_align:
            with open(osp.join(prox_base, f'alignment/{seq[:-9]}.npz'), 'rb') as f:
                aRt = np.load(f)
                aR = torch.from_numpy(aRt['R'])
                at = torch.from_numpy(aRt['t'])

        # seq_length = SEQ_LENGTHS[seq] - TRIM_EDGES * 2
        dekr_path = Path(f"/home/nugrinovic/code/CVPR_2024/EmbodiedPose/sample_data/"
                         f"dekr_kpts/prox/{seq_name}/dekr_preds.pkl")
        try:
            data = read_pickle(dekr_path)
        except:
            continue

        seq_length = data.shape[0]

        cur_subj_id = seq.split('_')[1]
        gender = 'female' if int(cur_subj_id) in FEMALE_SUBJ_IDS else 'male'

        smpl_dict = {'pose': np.zeros((seq_length, 72)), 'trans': np.zeros((seq_length, 3)),
                     'shape': np.zeros((seq_length, 16)), 'joints2d': np.zeros((seq_length, 25, 3)),
                     'gender': gender}

        # for i, (motion_file) in enumerate(motion_files):
        motion = read_pickle(motion_files[0])

        # seq_length = motion['root_orient']
        smpl_dict['pose'][:, :3] = motion['root_orient']
        smpl_dict['pose'][:, 3:66] = motion['pose_body']
        smpl_dict['trans'][:] = motion['trans']
        smpl_dict['shape'][:][:, :10] = motion['betas']  # Betas!!
        # Transform from the camera to world coordinate system
        pose, trans = transform_smpl(smpl_dict, R, t, seq_length, offset=humanoid.model.body_pos[1])
        smpl_dict['pose'][:, :3] = pose
        smpl_dict['trans'] = trans

        if 0:
            from utils.smpl import smpl_to_verts
            from utils.misc import save_trimesh
            pose = smpl_dict['pose']
            trans = smpl_dict['trans']
            verts, faces = smpl_to_verts(pose, trans)
            save_trimesh(verts[0, 0], faces, "inspect_out/prox/meshes/process/pose.ply")


        # Align the ground plane to the xy plane of the world coordinate system
        if not args.ignore_align:
            pose, trans = transform_smpl(smpl_dict, aR, at, seq_length, offset=humanoid.model.body_pos[1])
            smpl_dict['pose'][:, :3] = pose
            smpl_dict['trans'] = trans

        if 0:
            from utils.smpl import smpl_to_verts
            from utils.misc import save_trimesh
            pose = smpl_dict['pose']
            trans = smpl_dict['trans']
            verts, faces = smpl_to_verts(pose, trans)
            save_trimesh(verts[0, 0], faces, "inspect_out/prox/meshes/process/pose_aligned.ply")

        # Align the ground plane to the xy plane of the world coordinate system
        if args.noisy_rotz:
            trans_dummy = torch.zeros_like(t)
            if args.noisy_rotx:
                noisyR = torch.from_numpy(Tz[:3, :3] @ Tx[:3, :3])
            else:
                noisyR = torch.from_numpy(Tz[:3, :3])
            pose, trans = transform_smpl(smpl_dict, noisyR, trans_dummy, seq_length, offset=humanoid.model.body_pos[1])
            smpl_dict['pose'][:, :3] = pose
            smpl_dict['trans'] = trans
            if 0:
                from utils.smpl import smpl_to_verts
                from utils.misc import save_trimesh
                pose = smpl_dict['pose']
                trans = smpl_dict['trans']
                verts, faces = smpl_to_verts(pose, trans)
                # save_trimesh(verts[0, 0], faces, "inspect_out/prox/meshes/process/pose_noisy_rot.ply")
                save_trimesh(verts[0, 0], faces, "inspect_out/prox/meshes/process/pose_noisy_rot_pid4.ply")

        # NUK: added by me to make the floor z=0
        if args.to_floor:
            pose = smpl_dict['pose']
            trans = smpl_dict['trans']
            verts, faces = smpl_to_verts(pose, trans)
            mesh = trimesh.Trimesh(vertices=verts[0, 0].detach().cpu().numpy(), faces=faces)
            bbox = mesh.bounding_box.bounds
            min_xyz = bbox[0]
            trans_floor_z = trans[:, 2] - min_xyz[None, 2]
            new_trans = np.concatenate([trans[:, :2], trans_floor_z[:, None]], 1)
            # verts, faces = smpl_to_verts(pose[0, None], new_trans[0, None])
            # save_trimesh(verts[0, 0], faces, "inspect_out/prox/meshes/process/prox_floor.ply")
            smpl_dict['trans'] = new_trans

        if args.joints_source == 'dekr':
            dekr17_to_op_map = [0, 0, 6, 8, 10, 5, 7, 9, 0, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
            dekr_path = Path(f"/home/nugrinovic/code/CVPR_2024/EmbodiedPose/sample_data/"
                             f"dekr_kpts/prox/{seq_name}/dekr_preds.pkl")
            try:
                data = read_pickle(dekr_path)
            except:
                print(f"Failed to load {dekr_path}")
                continue
            op_kpts = data[:, dekr17_to_op_map]
            op_kpts[:, 1] = (op_kpts[:, 2] + op_kpts[:, 5]) / 2
            op_kpts[:, 8] = (op_kpts[:, 9] + op_kpts[:, 12]) / 2
            smpl_dict['joints2d'][:, :19] = op_kpts[:seq_length]

            if 0:
                image_folder = Path(
                    "/home/nugrinovic/code/CVPR_2024/EmbodiedPose/data/prox/recordings/N0Sofa_00145_01/Color_flipped")
                img_files = sorted(list(image_folder.glob("*.jpg")))
                idx = 0
                img = read_image_PIL(img_files[idx])
                plot_joints_cv2(img, op_kpts[0, None, :, :2], show=True, with_text=True)

        elif args.joints_source == 'smpl_gt':
            dekr_path = Path(f"/home/nugrinovic/code/CVPR_2024/EmbodiedPose/data/prox_embodied/{seq_name}/smpl_jts2d.pkl")
            op_kpts = read_pickle(dekr_path)
            smpl_dict['joints2d'][:, :15] = op_kpts[:seq_length]
        else:
            smpl_dict['joints2d'][:] = obs["joints2d"]

        if args.noisy_2d:
            # joints2d are in pixel space
            j2d = smpl_dict['joints2d'] # (N, 25, 3)
            N = j2d.shape[0]
            mu, sigma = 0, args.noisy_2d  # mean and standard deviation
            s = np.random.normal(mu, sigma, N*19*2)
            s = s.reshape(N, 19, 2)
            smpl_dict['joints2d'][:, :19, :2] = j2d[:, :19, :2] + s
        if args.set_conf_ones:
            j2d = smpl_dict['joints2d'] # (N, 25, 3)
            smpl_dict['joints2d'][:, :19, 2] = 1.0

        # override joints2d with author data
        if args.override_2d:
            au_path = "/home/nugrinovic/code/CVPR_2024/EmbodiedPose/sample_data/thirdeye_anns_prox_only_one_trunc.pkl"
            author_data = joblib.load(au_path)["N0Sofa_00145_01"]
            joints2d = author_data["joints2d"]
            smpl_dict['joints2d'] = joints2d[:seq_length]

        #NUK: it seems that this flips the OP keypoints because the orignal Prox imgs are flipped
        # they must have estimated the kpts w/out flipping the image, because I flip the image first
        # I don't need to use this
        # kp2d_files = sorted(glob.glob(osp.join(prox_base, "keypoints", seq, "*")))
        # kp_data = np.array([read_keypoints(kp2d_file) for kp2d_file in kp2d_files[TRIM_EDGES:-TRIM_EDGES]])
        if args.flip_joints:
            kp_data = obs["joints2d"]
            kp_data = kp_data[:, OP_FLIP_MAP, :]
            kp_data[:, :, 0] = IMG_WIDTH - kp_data[:, :, 0]
            smpl_dict['joints2d'] = kp_data

        entry = smpl_2_entry(args, seq, smpl_dict)

        if args.debug:
            count += 1
            debug_frames = args.debug_frames
            excluded = ["betas", "cam", "points3d", "gender", "seq_name"]
            for k, v in entry.items():
                if k in excluded:
                    continue
                entry[k] = v[:debug_frames]
                # print(k, entry[k].shape)

            # if count > 5:
            #     break

        data_res[seq] = entry

        if 0:
            from utils.smpl import smpl_to_verts
            from utils.misc import save_trimesh
            pose = smpl_dict['pose']
            trans = smpl_dict['trans']
            verts, faces = smpl_to_verts(pose, trans)
            save_trimesh(verts[0, 0], faces, "inspect_out/prox/meshes/process/prox_gt_aligned.ply")


    x_ang_d = int(np.degrees(x_angle))
    z_ang_d = int(np.degrees(z_angle))
    print(data_res.keys())
    out_base = f'/home/nugrinovic/code/CVPR_2024/EmbodiedPose/sample_data/prox/'

    flipped_name = "_flipped" if args.flip_joints else ""
    override_name = "_author2dkpts" if args.override_2d else ""
    joints_name = f"_{args.joints_source}" if args.joints_source else ""
    aligned_name = "_noalign" if args.ignore_align else ""
    floor_name = "_floor" if args.to_floor else ""
    noisy_name = f"_noisyZ{z_ang_d}" if args.noisy_rotz else ""
    noisyX_name = f"_noisyX{x_ang_d}" if args.noisy_rotx else ""
    confs_name = "_confsONE" if args.set_conf_ones else ""
    noisy2d_name = f"_noisy2d{int(sigma)}" if args.noisy_2d else ""
    debug_name = "_debug" if args.debug else ""
    debug_fr_name = f"_{debug_frames}" if args.debug else ""
    no_scene_name = "_no_scene" if not args.scene else ""
    output_file_name = osp.join(out_base,
                                f'thirdeye_anns_all_{args.dataset}{flipped_name}'
                                f'{override_name}{joints_name}{aligned_name}{floor_name}{noisy_name}'
                                f'{noisyX_name}{confs_name}{noisy2d_name}{debug_name}{debug_fr_name}{no_scene_name}.pkl')
    print(output_file_name, len(data_res))
    Path(output_file_name).parent.mkdir(exist_ok=True, parents=True)
    joblib.dump(data_res, open(output_file_name, "wb"))
