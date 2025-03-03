import torch
from pyquaternion import Quaternion as Q
from utils.misc import read_pickle
from utils.smpl import smpl_to_verts
from scipy.spatial.transform import Rotation as sRot
from utils.misc import save_mesh
import numpy as np




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
        final_Rt = Tz180[None] @ Txm90[None] @ Rt_gnd_inv[None] #@ T_c2w
    else:
        final_Rt = Txm90[None] @ Rt_gnd_inv[None]  # @ T_c2w
    scene_dict['final_Rt'] = final_Rt
    return scene_dict


def rot_pose(pose_aa, final_Rt):
    all = False
    pose_aa_i = pose_aa.copy()
    if len(pose_aa_i.shape) == 3:
        B, T, _ = pose_aa.shape
        pose_aa_i = pose_aa_i.reshape(-1, 72)
        all = True
    ori = sRot.from_rotvec(pose_aa_i[:, :3]).as_matrix()
    new_root = final_Rt[:, :3, :3] @ ori
    new_root = sRot.from_matrix(new_root).as_rotvec()
    pose_aa_i[:, :3] = new_root
    if all:
        pose_aa_i = pose_aa_i.reshape(B, T, 72)
    return pose_aa_i


def rot_and_correct_smplx_offset_full(pose_aa, trans_i_in, betas, final_Rt, get_verts=False):

    pose_aa_r = rot_pose(pose_aa, final_Rt)


    trans = trans_i_in.copy()
    trans_all, v_all = [], []

    for n in range(len(pose_aa)):
        trans_i = trans[n]
        pose_aa_i = pose_aa[n]
        betas_i = betas[n]
        (_, joints), _ = smpl_to_verts(pose_aa_i[None], trans_i[None], betas=betas_i[None], return_joints=True)
        joints = joints.numpy()
        pelvis = joints[0, :, 0, None] # (B, 1, 3)
        trans_i = trans_i[None, None]
        pelvis = pelvis - trans_i
        trans_i = (final_Rt[:, :3, :3] @ (trans_i + pelvis).transpose(0, 2, 1)).transpose(0, 2, 1) + final_Rt[:, None, :3, 3] - pelvis
        trans_i = trans_i[:, 0]
        if get_verts:
            verts, faces = smpl_to_verts(pose_aa_i[None], trans_i, betas=betas_i[None], return_joints=False)
            v_all.append(verts)
        trans_all.append(trans_i)
    trans_all = np.stack(trans_all)

    debug = False
    if debug:
        from utils.misc import save_trimesh
        from pathlib import Path
        op = "inspect_out/hi4d/smpl_rot/"
        Path(op).mkdir(parents=True, exist_ok=True)
        verts, faces = smpl_to_verts(pose_aa[0, None], trans_i_in[0, None], betas=betas[0, None])
        save_trimesh(verts[0, 0], faces, f"{op}/orig_pose.ply")
        verts, faces = smpl_to_verts(pose_aa_r[0, None], trans_all[0], betas=betas[0, None])
        save_trimesh(verts[0, 0], faces, f"{op}/rot_pose.ply")

    if get_verts:
        v_all = np.concatenate(v_all)
        return pose_aa_r, trans_all, v_all, faces
    return pose_aa_r, trans_all


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



