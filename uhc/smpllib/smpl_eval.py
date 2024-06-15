# import glob
import os
import sys
# import pdb
# import os.path as osp
import torch

sys.path.append(os.getcwd())

# import numpy as np

# import torch
# import numpy as np
# import pickle as pk
# from tqdm import tqdm
# from collections import defaultdict
# import random
# import argparse

# from uhc.utils.transformation import euler_from_quaternion, quaternion_matrix
from uhc.utils.math_utils import *
# from uhc.smpllib.smpl_mujoco import smpl_to_qpose, qpos_to_smpl
import copy

from utils.misc import plot_joints_cv2
from utils.misc import read_image_PIL

def p_mpjpe(predicted, target, return_aligned=False, return_transform=False):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY # Scale
    t = muX - a*np.matmul(muY, R) # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a*np.matmul(predicted, R) + t

    # Return MPJPE
    pa_mpjpe = np.linalg.norm(predicted_aligned - target, axis=len(target.shape)-1)
    if return_aligned:
        return pa_mpjpe, predicted_aligned
    elif return_transform:
        return a, R, t
    return pa_mpjpe


def compute_metrics(res, converter=None, key=None, args=None, n=None, use_wbpos=False, use_smpl_gt=False):
    debug = False
    res = copy.deepcopy(res)
    res_dict = {}

    # if use_wbpos:
    #     jpos_pred = (converter.jpos_new_2_smpl(res["pred_jpos_wbpos"])
    #                  if converter is not None else res["pred_jpos_wbpos"])
    jpos_pred = (converter.jpos_new_2_smpl(res["pred_jpos"])
                 if converter is not None else res["pred_jpos"])

    jpos_gt = (converter.jpos_new_2_smpl(res["gt_jpos"])
               if converter is not None else res["gt_jpos"])
        
    # res["pred"] and res["gt"] are in the (B, 76) EmbPose format
    traj_pred = res["pred"]
    traj_gt = res["gt"]

    batch_size = traj_pred.shape[0]
    batch_size_gt = jpos_gt.shape[0]
    if batch_size_gt != batch_size:
        print(f"warning: batch size mismatch bwt pred and GT! batch_size_gt:{batch_size_gt}, batch_size: {batch_size}")
        print(f"WARNING: taking the min batch size")
        batch_size = min(batch_size, batch_size_gt)
    # there are joints
    jpos_pred = jpos_pred[:batch_size].reshape(batch_size, -1, 3)
    jpos_gt = jpos_gt[:batch_size].reshape(batch_size, -1, 3)

    root_mat_pred = get_root_matrix(traj_pred)[:batch_size]
    root_mat_gt = get_root_matrix(traj_gt)[:batch_size]
    root_dist = get_frobenious_norm(root_mat_pred, root_mat_gt) * 1000
    # these take joints as input
    # here jpos_gt and jpos_pred are in the (B, 24, 3) format
    if 'gt_jpos_vis' in res:
        vis = res['gt_jpos_vis'].astype(bool)
        jpos_pred_v = jpos_pred[:, vis]
        jpos_gt_v = jpos_gt[:, vis]
    else:
        jpos_pred_v = jpos_pred.copy()
        jpos_gt_v = jpos_gt.copy()

    vel_dist = compute_error_vel(jpos_pred_v, jpos_gt_v).mean(axis = -1) * 1000
    accel_dist = compute_error_accel(jpos_pred_v, jpos_gt_v).mean(axis = -1) * 1000

    if key=='acro1_check-the-change1_cam20':
        print('here')

    accel_dist.mean()
    mpjpe_g = np.linalg.norm(jpos_pred - jpos_gt, axis=2).mean(axis=-1) * 1000

    jpos_pred_orig = jpos_pred.copy()
    jpos_gt_orig = jpos_gt.copy()

    if jpos_pred.shape[-2] == 24:
        jpos_pred = jpos_pred - jpos_pred[:, 0:1]  # zero out root
        jpos_gt = jpos_gt - jpos_gt[:, 0:1]
    elif jpos_pred.shape[-2] == 14:
        jpos_pred = jpos_pred - jpos_pred[..., 7:8, :]  # zero out root
        jpos_gt = jpos_gt - jpos_gt[..., 7:8, :]
    elif jpos_pred.shape[-2] == 12:
        jpos_pred = jpos_pred - jpos_pred[..., 7:8, :]  # zero out root
        jpos_gt = jpos_gt - jpos_gt[..., 7:8, :]

    if debug:
        from utils.misc import save_pointcloud
        # seq_name = "Grab_1"
        im_path = f"/sailhome/nugrinov/code/CVPR_2024/slahmr_release/slahmr/videos/chi3d/train/s02/images/{key}/000001.jpg"
        img = read_image_PIL(im_path)
        plot_joints_cv2(img, jpos_pred[0, None], show=True, with_text=True, sc=3)
        plot_joints_cv2(img, jpos_gt[0, None], show=True, with_text=True, sc=3)
        idx = 50
        save_pointcloud(jpos_pred[idx], f"inspect_out/eval/compute_metric/{key}/jpos_pred.ply")
        save_pointcloud(jpos_gt[idx], f"inspect_out/eval/compute_metric/{key}/jpos_gt.ply")
        # rotate 180 in z?

    if 'gt_jpos_vis' in res:
        vis = res['gt_jpos_vis'].astype(bool)
        jpos_pred = jpos_pred[:, vis]
        jpos_gt = jpos_gt[:, vis]

    pa_mpjpe, pred_pa = p_mpjpe(jpos_pred, jpos_gt, return_aligned=True)
    pa_mpjpe_per_jt = 1000* pa_mpjpe
    pa_mpjpe = pa_mpjpe.mean(axis=-1) * 1000
    mpjpe = np.linalg.norm(jpos_pred - jpos_gt, axis=2).mean(axis=-1) * 1000
    mpjpe_per_jt = np.linalg.norm(jpos_pred - jpos_gt, axis=2)* 1000
    # pa_mpjpe = p_mpjpe(jpos_pred, jpos_gt).mean(axis=-1) * 1000
    # mpjpe = np.linalg.norm(jpos_pred - jpos_gt, axis=2).mean(axis=-1) * 1000

    if debug:
        from utils.joints3d import joints_to_skel
        e_max = int(mpjpe.max())
        midx = mpjpe.argmax()

        op = f"inspect_out/metrics_pose/{args.data_name}/{args.model_type}-{args.exp_name}/{key}"
        format = 'smpl'
        joints_to_skel(jpos_pred[midx, None], f"{op}/skels_pred_{n}_{midx:03d}_{e_max}.ply", format=format, radius=0.015, sphere_rad=0.02, save_jts=True)
        joints_to_skel(jpos_gt[midx, None], f"{op}/skels_gt_{n}_{midx:03d}_{e_max}.ply", format=format, radius=0.015, sphere_rad=0.02, save_jts=True)
        idx = 66
        joints_to_skel(jpos_pred_orig[idx, None], f"{op}/skels_pred_ORI_{n}_{idx:03d}.ply", format='smpl', radius=0.015, sphere_rad=0.02)
        joints_to_skel(jpos_pred[idx, None], f"{op}/skels_pred_{n}_{idx:03d}.ply", format='smpl_red', radius=0.015, sphere_rad=0.02)
        joints_to_skel(pred_pa[idx, None], f"{op}/skels_pred_PA_{n}_{idx:03d}.ply", format='smpl_red', radius=0.015, sphere_rad=0.02)
        joints_to_skel(jpos_gt[idx, None], f"{op}/skels_gt_{n}_{idx:03d}.ply", format='smpl_red', radius=0.015, sphere_rad=0.02)
        joints_to_skel(jpos_gt_orig[idx, None], f"{op}/skels_gt_ORI_{n}_{idx:03d}.ply", format='smpl', radius=0.015, sphere_rad=0.02)


    succ = not res["fail_safe"] and res["percent"] == 1

    info = {}
    info["floor_z"] = 0
    res_dict["root_dist"] = root_dist
    res_dict["pa_mpjpe"] = pa_mpjpe
    res_dict["pa_mpjpe_per_jt"] = pa_mpjpe_per_jt
    res_dict["mpjpe_per_jt"] = mpjpe_per_jt
    res_dict["mpjpe"] = mpjpe
    res_dict["mpjpe_g"] = mpjpe_g
    res_dict["accel_dist"] = accel_dist
    res_dict["vel_dist"] = vel_dist
    res_dict["succ"] = np.array([succ])
    res_dict["jpos_pred"] = jpos_pred.reshape(batch_size, -1)
    res_dict["jpos_gt"] = jpos_gt.reshape(batch_size, -1)


    pred_sum = jpos_pred.sum()
    gt_sum = jpos_gt.sum()
    res_dict["pred_sum_check"] = pred_sum
    res_dict["gt_sum_check"] = gt_sum
    res_dict["n_frames"] = len(jpos_pred)

    # debug = True
    if debug:
        from utils.misc import save_pointcloud
        op = f"inspect_out/compute_metrics/{args.model_type}-{args.exp_name}/{key}"
        idx = 0
        save_pointcloud(jpos_gt[idx].reshape(-1, 3), f"{op}/joints-rel/{idx}/gt_jpos_{n}.ply")
        save_pointcloud(jpos_pred[idx].reshape(-1, 3), f"{op}/joints-rel/{idx}/pred_jpos_{n}-{mpjpe[idx]:.0f}.ply")
        save_pointcloud(pred_pa[idx].reshape(-1, 3), f"{op}/joints-rel/{idx}/pred_jpos_PA_{n}-{pa_mpjpe[idx]:.0f}.ply")

        from utils.smpl import smpl_to_verts
        from utils.smpl import save_mesh
        pose_aa_i = res["smpl_pred"]["pose_aa"]
        trans_i = res["smpl_pred"]["trans"]
        betas = res["smpl_pred"]["betas"]
        verts, faces = smpl_to_verts(pose_aa_i, trans_i, betas=betas, return_joints=False)
        save_mesh(verts[0, idx, None], faces=faces, out_path=f"{op}/meshes/{idx}/pred_smpl_{n}.ply")

    return res_dict

def compute_phys_metrics(res, converter=None, key=None, args=None, n=None, use_wbpos=False, use_smpl_gt=False):
    res_dict = {}
    info = {}
    # 5mm tolerance as in PhysDiff
    info["floor_z"] = -0.005
    if "pred_vertices" in res:
        pred_vertices = torch.from_numpy(res["pred_vertices"]).float()
        pred_vertices = pred_vertices[1:]
        # mean distance in mm for all verts that have a height < 0
        pent_frames = compute_ground_penetration(pred_vertices, info)
        pent = np.mean(pent_frames)
        skate = np.mean(compute_skate(pred_vertices, info))

        float_frames = compute_floating(pred_vertices, info)
        float = np.mean(float_frames)


        res_dict["pentration"] = pent
        res_dict["skate"] = skate
        res_dict["float"] = float

        del res["pred_vertices"]
        debug = False
        if debug:
            from utils.misc import save_pointcloud
            op = f"inspect_out/compute_metrics/{args.metric_type}/{args.model_type}-{args.exp_name}/{key}"
            idx = 0
            save_pointcloud(pred_vertices[idx], f"{op}/verts_pred_{n}_{idx:03d}.ply")
    return res_dict


def compute_ground_penetration(vert, info):
    pen = []
    for vert_i in vert:
        vert_z = vert_i[:, 2] - info["floor_z"]
        pind = vert_z < 0
        # 5mm tolerance as in PhysDiff
        # pind = vert_z < 5
        if torch.any(pind):
            pen_i = -vert_z[pind].mean().item() * 1000
        else:
            pen_i = 0.0
        pen.append(pen_i)
    return pen


def compute_floating(vert, info):
    flo = []
    for vert_i in vert:
        min_z = min(vert_i[:, 2])
        if min_z > 0:
            float = min_z * 1000
        else:
            float = 0.0
        flo.append(float)
    return flo

def compute_skate(vert, info):
    skate = []
    for t in range(vert.shape[0] - 1):
        # which vertices penetrates the floor in both frames?
        cind = (vert[t, :, 2] <= info["floor_z"]) & (vert[t + 1, :, 2] <=info["floor_z"])
        if torch.any(cind):
            # if any pair of vertices penetrates the floor, compute the average of the magnitude of the displacement
            offset = vert[t + 1, cind, :2] - vert[t, cind, :2]
            skate_i = torch.norm(offset, dim=1).mean().item() * 1000
        else:
            skate_i = 0.0
        skate.append(skate_i)
    return skate


def get_root_matrix(poses):
    matrices = []
    for pose in poses:
        mat = np.identity(4)
        root_pos = pose[:3]
        root_quat = pose[3:7]
        mat = quaternion_matrix(root_quat)
        mat[:3, 3] = root_pos
        matrices.append(mat)
    return matrices


def get_joint_vels(poses, dt):
    vels = []
    for i in range(poses.shape[0] - 1):
        v = get_qvel_fd(poses[i], poses[i + 1], dt, "heading")
        vels.append(v)
    vels = np.vstack(vels)
    return vels


def get_joint_accels(vels, dt):
    accels = np.diff(vels, axis=0) / dt
    accels = np.vstack(accels)
    return accels


def get_frobenious_norm(x, y):
    error = []
    for i in range(len(x)):
        x_mat = x[i]
        y_mat_inv = np.linalg.inv(y[i])
        error_mat = np.matmul(x_mat, y_mat_inv)
        ident_mat = np.identity(4)
        error.append( np.linalg.norm(ident_mat - error_mat, "fro"))
    return np.array(error) / len(x)


def get_mean_dist(x, y):
    return np.linalg.norm(x - y, axis=1).mean()


def get_mean_abs(x):
    return np.abs(x).mean()


def compute_accel(joints):
    """
    Computes acceleration of 3D joints.
    Args:
        joints (Nx25x3).
    Returns:
        Accelerations (N-2).
    """
    velocities = joints[1:] - joints[:-1]
    acceleration = velocities[1:] - velocities[:-1]
    acceleration_normed = np.linalg.norm(acceleration, axis=2)
    return acceleration_normed


def compute_error_accel(joints_gt, joints_pred, vis=None):
    """
    Computes acceleration error:
        1/(n-2) \sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
    Note that for each frame that is not visible, three entries in the
    acceleration error should be zero'd out.
    Args:
        joints_gt (Nx14x3).
        joints_pred (Nx14x3).
        vis (N).
    Returns:
        error_accel (N-2).
    """
    # (N-2)x14x3
    accel_gt = joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
    accel_pred = joints_pred[:-2] - 2 * joints_pred[1:-1] + joints_pred[2:]

    normed = np.linalg.norm(accel_pred - accel_gt, axis=2)

    if vis is None:
        new_vis = np.ones(len(normed), dtype=bool)
    else:
        invis = np.logical_not(vis)
        invis1 = np.roll(invis, -1)
        invis2 = np.roll(invis, -2)
        new_invis = np.logical_or(invis, np.logical_or(invis1, invis2))[:-2]
        new_vis = np.logical_not(new_invis)

    return normed[new_vis]


def compute_vel(joints):
    velocities = joints[1:] - joints[:-1]
    velocity_normed = np.linalg.norm(velocities, axis=2)
    return velocity_normed


def compute_error_vel(joints_gt, joints_pred, vis=None):
    vel_gt = joints_gt[1:] - joints_gt[:-1]
    vel_pred = joints_pred[1:] - joints_pred[:-1]
    normed = np.linalg.norm(vel_pred - vel_gt, axis=2)

    if vis is None:
        new_vis = np.ones(len(normed), dtype=bool)
    return normed[new_vis]