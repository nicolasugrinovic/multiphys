import torch

from uhc.smpllib.smpl_robot import Robot
from uhc.smpllib.torch_smpl_humanoid import Humanoid
import mujoco_py
from uhc.utils.config_utils.copycat_config import Config as CC_Config
import os.path as osp
from utils.pyquaternion import Quaternion as Q
from utils.misc import read_pickle
from utils.smpl import smpl_to_verts

from utils.net_utils import get_hostname

from scipy.spatial.transform import Rotation as sRot
from utils.misc import save_mesh

torch.set_default_dtype(torch.float32)

hostname = get_hostname()


def load_humanoid():
    print("Loading humanoid...")
    # cc_cfg = CC_Config(cfg_id="copycat_e_1", base_dir="./")
    cc_cfg = CC_Config(cfg_id="copycat_eval_1", base_dir="./")
    smpl_robot = Robot(
        cc_cfg.robot_cfg,
        data_dir=osp.join(cc_cfg.base_dir, "data/smpl"),
        masterfoot=cc_cfg.masterfoot,
    )
    model = mujoco_py.load_model_from_xml(smpl_robot.export_xml_string().decode("utf-8"))
    humanoid = Humanoid(model=model)
    return smpl_robot, humanoid, cc_cfg


def get_smplx_flip_params():
    root_body_flip_perm = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20]
    jaw_eye_flip_perm = [22, 24, 23]
    hand_flip_perm = list(range(25 + 15, 25 + 2 * 15)) + list(range(25, 25 + 15))
    trans_flip_perm = [55]
    ll = root_body_flip_perm + jaw_eye_flip_perm + hand_flip_perm + trans_flip_perm
    flip_perm = []
    flip_inv = []
    for i in ll:
        flip_perm.append(3 * i)
        flip_perm.append(3 * i + 1)
        flip_perm.append(3 * i + 2)
        if i == 55:
            # trans only
            flip_inv.extend([-1., 1., 1.])
        else:
            # smplx jts
            flip_inv.extend([1., -1., -1.])
    flip_inv = torch.Tensor(flip_inv).reshape(1, -1)
    return flip_perm, flip_inv


def get_smpl_flip_params():
    pose_flip_perm = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22]
    trans_flip_perm = [24]
    ll = pose_flip_perm + trans_flip_perm
    flip_perm = []
    flip_inv = []
    for i in ll:
        flip_perm.append(3 * i)
        flip_perm.append(3 * i + 1)
        flip_perm.append(3 * i + 2)
        if i == 24:
            # trans only
            flip_inv.extend([-1., 1., 1.])
        else:
            # smplx jts
            flip_inv.extend([1., -1., -1.])
    flip_inv = torch.Tensor(flip_inv).reshape(1, -1)

    return flip_perm, flip_inv


def get_camera_transform(vid_name, res_dir, rot_180z=False):
    """
    Get the camera transform from slahmr and convert the poses to the same coordinate system.
    Note, body poses from slahmr are already in "world" coordinates, but with an estimated ground plane
    rotation and translation. Therefore, the only transformation necessary is the inverse of this ground Rt
    """
    scene_dict = read_pickle(f"{res_dir}/{vid_name}_scene_dict.pkl")
    Rt_gnd = scene_dict["ground"]
    # this is T_c2w from slahmr
    T_c2w = scene_dict["cameras"]["src_cam"]
    # this is the ground transform from slahmr, inverted as we want to take the poses to xy, z=0
    Rt_gnd_inv = torch.linalg.inv(Rt_gnd).to(T_c2w.device)
    # rotate by -90 deg around x to match the +z up camera from EmbodiedPose
    Txm90 = Q(axis=[1, 0, 0], angle=-np.pi / 2).transformation_matrix.astype(np.float32)
    Txm90 = torch.tensor(Txm90).float().to(T_c2w.device)
    Tz180 = Q(axis=[0, 0, 1], angle=np.pi).transformation_matrix.astype(np.float32)
    Tz180 = torch.from_numpy(Tz180).float().to(T_c2w.device)
    # T_c2w contains cameras for each frame, use this for dynamic camera
    # NOTE: we can ignore T_c2w transformation because slahmr already has it in world coordinates
    if rot_180z:
        final_Rt = Tz180[None] @ Txm90[None] @ Rt_gnd_inv[None] # @ T_c2w
    else:
        final_Rt = Txm90[None] @ Rt_gnd_inv[None]  # @ T_c2w
    scene_dict['final_Rt'] = final_Rt
    return scene_dict


def rot_pose(pose_aa, final_Rt):
    """
    pose_aa: can be either numpy or torch tensor
    """
    all = False

    if isinstance(pose_aa, torch.Tensor):
        pose_aa_i = pose_aa.clone()
    else:
        pose_aa_i = pose_aa.copy()

    if len(pose_aa_i.shape) == 3:
        B, T, _ = pose_aa.shape
        pose_aa_i = pose_aa_i.reshape(-1, 72)
        all = True

    if isinstance(final_Rt, torch.Tensor):
        ori = torch.tensor(sRot.from_rotvec(pose_aa_i[:, :3]).as_matrix()).float().to(final_Rt)
    else:
        ori = torch.tensor(sRot.from_rotvec(pose_aa_i[:, :3]).as_matrix()).float()
        final_Rt = torch.from_numpy(final_Rt)

    new_root = final_Rt[:, :3, :3] @ ori
    new_root = sRot.from_matrix(new_root.cpu()).as_rotvec()
    pose_aa_i[:, :3] = torch.tensor(new_root).float()
    if all:
        pose_aa_i = pose_aa_i.reshape(B, T, 72)
    return pose_aa_i


def rot_and_correct_smplx_offset(pose_aa, trans_i_in, betas, final_Rt, get_verts=False):
    trans = trans_i_in.clone()
    trans_all = []
    v_all = []
    # f_all = []
    for n in range(len(pose_aa)):
        trans_i = trans[n]
        pose_aa_i = pose_aa[n]
        betas_i = betas[n]
        (_, joints), _ = smpl_to_verts(pose_aa_i, trans_i, betas=betas_i, return_joints=True)
        pelvis = joints[0, :, 0, None] # (B, 1, 3)
        trans_i = trans_i[:, None]
        pelvis = pelvis.to(trans_i) - trans_i
        trans_i = (final_Rt[:, :3, :3] @ (trans_i + pelvis).permute(0, 2, 1)).permute(0, 2, 1) + final_Rt[:, None, :3, 3] - pelvis
        trans_i = trans_i[:, 0]
        if get_verts:
            verts, faces = smpl_to_verts(pose_aa_i, trans_i, betas=betas_i, return_joints=False)
            v_all.append(verts)
        trans_all.append(trans_i)
    trans_all = torch.stack(trans_all)
    if get_verts:
        v_all = torch.cat(v_all)
        return trans_all, v_all, faces
    return trans_all



def rot_and_correct_smplx_offset_full(pose_aa, trans_i_in, betas, final_Rt):
    """
    pose_aa: (B, 72)
    trans_i_in: (B, 3)
    betas: (B, 10)
    final_Rt: (1, 4, 4) ideally a pyquaternion transform

    returns:
    pose_aa_r: (B, 72)
    trans: (B, 3)
    """
    trans = trans_i_in.copy()

    pose_aa_r = rot_pose(pose_aa, final_Rt)

    (_, joints), _ = smpl_to_verts(pose_aa, trans, betas=betas, return_joints=True)
    joints = joints.numpy()
    pelvis = joints[0, :, 0, None]  # (B, 1, 3)
    trans = trans[:, None]  # (B, 1, 3)
    pelvis = pelvis - trans
    trans = (final_Rt[:, :3, :3] @ (trans + pelvis).transpose(0, 2, 1)).transpose(0, 2, 1) + final_Rt[:, :3, 3] - pelvis
    trans = trans[:, 0]

    return pose_aa_r, trans

def save_multi_mesh_sequence(verts_all, faces, mesh_folder, name='', debug_idx=None, ds=1):
    """ accepts a list of verts of dim [1, B, Vnum, 3], each element corresponds to one person
    this is meant to save meshes contain two people
    """
    pred_verts = torch.cat(verts_all, axis=0).cpu().numpy()
    pred_verts = np.transpose(pred_verts, (1, 0, 2, 3))
    for ni, p_verts in enumerate(pred_verts[::ds]):
        if debug_idx is not None:
            if ni != debug_idx:
                continue
        save_mesh(p_verts, faces=faces, out_path=f"{mesh_folder}/verts_{ni:04d}{name}.ply")
    print(f"Saved {len(pred_verts)} meshes to {mesh_folder}")


def read_smpl_meshes(data_chi3d_this, seq_len):
    root_orient = data_chi3d_this["root_orient"]
    pose_body = data_chi3d_this["pose_body"]
    trans = data_chi3d_this["trans"]
    betas = data_chi3d_this["betas"]
    pose_aa = torch.cat([root_orient[:seq_len], pose_body[:seq_len], torch.zeros(seq_len, 6).float().to(root_orient)], dim=-1)
    verts, faces = smpl_to_verts(pose_aa, trans[:seq_len].float(), betas=betas[:1])
    return verts, faces


import numpy as np


def main():
    pass


if __name__ == "__main__":
    main()
