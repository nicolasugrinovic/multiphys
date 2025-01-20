from utils.body_model import pose_to_vertices as pose_to_vertices_
import smplx
from functools import partial
import torch
import numpy as np
from utils.misc import save_trimesh
from uhc.smpllib.smpl_mujoco import qpos_to_smpl
from utils.process_utils import load_humanoid
from utils.misc import save_mesh

local_bm = smplx.create("data", 'smpl', use_pca=False, batch_size=1)
pose_to_vertices = partial(pose_to_vertices_, pose_type="smpl", alpha=1, bm=local_bm)

smpl_robot, humanoid, cc_cfg = load_humanoid()

def smpl_to_verts(humor_first_pose, init_trans, betas=None, return_joints=False, device='cpu'):
    """
    the input to smpl_to_verts is a (B, 72) vector of pose parameters,
    where B is the batch size
    init_trans is a (B, 3) vector of translation parameters
    they have to be numpy arrays
    """
    if isinstance(humor_first_pose, np.ndarray):
        humor_first_pose = torch.tensor(humor_first_pose).float()#.cuda()
    if isinstance(init_trans, np.ndarray):
        init_trans = torch.tensor(init_trans).float()#.cuda()
    # pose = np.concatenate([humor_first_pose, init_trans], axis=1)
    humor_first_pose = humor_first_pose.cpu()
    init_trans = init_trans.cpu()
    if betas is not None:
        if isinstance(betas, np.ndarray):
            betas = torch.tensor(betas).float()  # .cuda()
        betas = betas.cpu()

    pose = torch.cat([humor_first_pose, init_trans], axis=1)
    # pose = torch.from_numpy(pose).float()#.cuda()
    with torch.no_grad():
        verts = pose_to_vertices(pose[None], betas=betas, return_joints=return_joints)
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


def from_qpos_to_verts_w_model(qpos, model, betas=None, get_verts=False, ret_body_pose=False):

    # smpl_robot.load_from_skeleton(torch.from_numpy(betas[None,]), gender=[0], objs_info=None)
    # model = mujoco_py.load_model_from_xml(smpl_robot.export_xml_string().decode("utf-8"))

    pose_aa_tr, trans_tr, body_pos = qpos_to_smpl(qpos, model, cc_cfg.robot_cfg.get("model", "smpl"), ret_body_pose=True)
    pose_aa_tr = pose_aa_tr.reshape([pose_aa_tr.shape[0], -1])

    if get_verts:
        verts, faces = smpl_to_verts(pose_aa_tr, trans_tr, betas=betas, return_joints=False)
        return pose_aa_tr, trans_tr, verts, faces
    if ret_body_pose:
        return pose_aa_tr, trans_tr, body_pos
    return pose_aa_tr, trans_tr
