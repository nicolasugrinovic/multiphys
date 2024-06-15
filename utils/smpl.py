from utils.body_model import pose_to_vertices as pose_to_vertices_
import smplx
from functools import partial
import torch
import numpy as np
from utils.misc import save_trimesh
from utils.misc import save_mesh

local_bm = smplx.create("data", 'smpl', use_pca=False, batch_size=1)
pose_to_vertices = partial(pose_to_vertices_, pose_type="smpl", alpha=1, bm=local_bm)


def smpl_to_verts_bs(humor_first_pose, init_trans, betas=None, return_joints=False, w_grad=False, body_model=None,
                     batch_size=1):
    """
    humor_first_pose: (B, T, 72) can be either numpy or torch tensor
    init_trans: (B, T, 3) can be either numpy or torch tensor

    the input to smpl_to_verts is a (B, 72) vector of pose parameters,
    where B is the batch size
    init_trans is a (B, 3) vector of translation parameters
    they have to be numpy arrays
    """

    local_bm_bs = smplx.create("data", 'smpl', use_pca=False, batch_size=batch_size)
    pose_to_vertices = partial(pose_to_vertices_, pose_type="smpl", alpha=1, bm=local_bm_bs)

    if body_model is not None:
        pose_to_vertices_this = partial(pose_to_vertices_, pose_type="smpl", alpha=1, bm=body_model)
    else:
        body_model = local_bm_bs
        pose_to_vertices_this = pose_to_vertices

    if isinstance(humor_first_pose, np.ndarray):
        humor_first_pose = torch.tensor(humor_first_pose).float()#.cuda()
    if isinstance(init_trans, np.ndarray):
        init_trans = torch.tensor(init_trans).float()#.cuda()
    # pose = np.concatenate([humor_first_pose, init_trans], axis=1)
    device = body_model.transl.device
    if device.type == 'cpu':
        humor_first_pose = humor_first_pose.cpu()
        init_trans = init_trans.cpu()
    if betas is not None:
        if isinstance(betas, np.ndarray):
            betas = torch.tensor(betas).float()  # .cuda()
        if device.type == 'cpu':
            betas = betas.cpu()

    pose = torch.cat([humor_first_pose, init_trans], axis=-1)
    # pose = torch.from_numpy(pose).float()#.cuda()
    if w_grad:
        verts = pose_to_vertices_this(pose, betas=betas, return_joints=return_joints)
    else:
        with torch.no_grad():
            verts = pose_to_vertices_this(pose, betas=betas, return_joints=return_joints)
    return verts, local_bm.faces


def smpl_to_verts(humor_first_pose, init_trans, betas=None, return_joints=False, device='cpu', w_grad=False, body_model=None):
    """
    humor_first_pose: (B, 72) can be either numpy or torch tensor
    init_trans: (B, 3) can be either numpy or torch tensor

    the input to smpl_to_verts is a (B, 72) vector of pose parameters,
    where B is the batch size
    init_trans is a (B, 3) vector of translation parameters
    they have to be numpy arrays
    """

    if body_model is not None:
        pose_to_vertices_this = partial(pose_to_vertices_, pose_type="smpl", alpha=1, bm=body_model)
    else:
        body_model = local_bm
        pose_to_vertices_this = pose_to_vertices

    if isinstance(humor_first_pose, np.ndarray):
        humor_first_pose = torch.tensor(humor_first_pose).float()#.cuda()
    if isinstance(init_trans, np.ndarray):
        init_trans = torch.tensor(init_trans).float()#.cuda()
    # pose = np.concatenate([humor_first_pose, init_trans], axis=1)
    device = body_model.transl.device
    if device.type == 'cpu':
        humor_first_pose = humor_first_pose.cpu()
        init_trans = init_trans.cpu()
    if betas is not None:
        if isinstance(betas, np.ndarray):
            betas = torch.tensor(betas).float()  # .cuda()
        if device.type == 'cpu':
            betas = betas.cpu()

    pose = torch.cat([humor_first_pose, init_trans], axis=1)
    # pose = torch.from_numpy(pose).float()#.cuda()
    if w_grad:
        verts = pose_to_vertices_this(pose[None], betas=betas, return_joints=return_joints)
    else:
        with torch.no_grad():
            verts = pose_to_vertices_this(pose[None], betas=betas, return_joints=return_joints)
    return verts, local_bm.faces


def from_qpos_to_smpl(pred_qpos, curr_env, betas=None, agent_id=None):
    # input has to be (76,)
    assert len(pred_qpos.shape) == 1, "input has to be (76,)"
    pred_smpl = curr_env.get_humanoid_pose_aa_trans(pred_qpos[None], agent_id=agent_id)
    pred_pose = pred_smpl[0].reshape([1, 72])
    pred_verts, faces = smpl_to_verts(pred_pose, pred_smpl[1], betas)
    return pred_verts, faces


def from_qpos_to_smpl_single(pred_qpos, curr_env, betas=None):
    # here curr_env is an env for single agent
    # input has to be (76,)
    assert len(pred_qpos.shape) == 1, "input has to be (76,)"
    pred_smpl = curr_env.get_humanoid_pose_aa_trans(pred_qpos[None])
    pred_pose = pred_smpl[0].reshape([1, 72])
    pred_verts, faces = smpl_to_verts(pred_pose, pred_smpl[1], betas)
    return pred_verts, faces

def from_qpos_to_verts_save(gt_qpos, curr_env, inspect_path, out_fname="verts.ply", agent_id=None):
    assert agent_id is not None, "agent_id has to be specified"
    # gt_qpos has to be: shape (76,)
    gt_verts, faces = from_qpos_to_smpl(gt_qpos, curr_env, agent_id=agent_id)
    save_trimesh(gt_verts[0, 0], faces, inspect_path + out_fname)


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