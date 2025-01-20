import os
import os.path as osp
import sys
sys.path.append(os.getcwd())
from uhc.smpllib.smpl_robot import Robot
from uhc.smpllib.torch_smpl_humanoid import Humanoid
import mujoco_py
from uhc.utils.config_utils.copycat_config import Config as CC_Config
import torch
import numpy as np
from embodiedpose.constants import LEFT_RIGHT_IDX
from uhc.utils.torch_geometry_transforms import (angle_axis_to_rotation_matrix)
from uhc.utils.transform_utils import (convert_aa_to_orth6d, rotation_matrix_to_angle_axis)
from uhc.smpllib.smpl_mujoco import smpl_to_qpose_torch
from embodiedpose.models.humor.utils.humor_mujoco import reorder_joints_to_humor
from embodiedpose.models.humor.utils.velocities import estimate_velocities
from uhc.utils.torch_ext import dict_to_numpy


def load_humanoid():
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

smpl_robot, humanoid, cc_cfg = load_humanoid()

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
    seq_name,
    smpl_dict,
    camera_pose,
    debug=False,
        rot_tr=True
):
    R = np.array(camera_pose["R"]).astype(np.float32) # (3, 3)
    tr = np.array(camera_pose["T"]).astype(np.float32).reshape(1, 3) # (1, 3)
    tr = tr[0]  # (3, )
    K_ = camera_pose["K"]
    K = np.eye(3).astype(np.float32) # (3, 3)
    K[0, 0] = K_["f"][0]
    K[1, 1] = K_["f"][1]
    K[0, 2] = K_["c"][0]
    K[1, 2] = K_["c"][1]
    # this is important!!
    R = R.T
    if rot_tr:
        tr = -np.matmul(R, tr)
    pose_aa = smpl_dict["pose"]
    trans = smpl_dict["trans"]
    seq_len = pose_aa.shape[0]
    shape = smpl_dict["shape"] if "shape" in smpl_dict else np.zeros([seq_len, 10])
    # gets the mean of the first 10 estimated bodies in the sequence
    mean_shape = shape[0:10].mean(axis=0) # shape is (B, 16)
    gender = smpl_dict["gender"] if "gender" in smpl_dict else "neutral"
    # joints --> shape (65, 25, 3), min: 0, max: 713.2
    joints2d = smpl_dict["joints2d"] if "joints2d" in smpl_dict else None
    points3d = smpl_dict["points3d"] if "points3d" in smpl_dict else None

    if debug:
        from utils.misc import plot_joints_cv2
        verts, faces = smpl_to_verts(pose_aa[0, None], trans[0, None])
        save_trimesh(verts[0, 0], faces, f"inspect_out/chi3d/meshes/process/floor{person_name}_entry.ply")
        # black = np.zeros([1080, 1920, 3], dtype=np.uint8)
        j2d = joints2d[0]
        black = np.zeros([900, 900, 3], dtype=np.uint8)
        plot_joints_cv2(black, joints2d[0, None],
                        return_img=False, with_text=True,
                        sc=3)

    seq_length = pose_aa.shape[0]
    if seq_length < 10:
        return None
    pose_aa = torch.from_numpy(pose_aa).float()
    pose_seq_6d = convert_aa_to_orth6d(pose_aa).reshape(-1, 144)
    smpl_robot.load_from_skeleton(torch.from_numpy(mean_shape[None,]), gender=[0], objs_info=None)
    model = mujoco_py.load_model_from_xml(smpl_robot.export_xml_string().decode("utf-8"))
    humanoid = Humanoid(model=model)
    # pose_aa is (B, 72), trans is (B, 3)
    qpos = smpl_to_qpose_torch(pose_aa, model, trans=torch.from_numpy(trans), count_offset=True)
    fk_result = humanoid.qpos_fk(qpos, to_numpy=False)

    root_orient = pose_aa[:, :3]
    joints_humor = reorder_joints_to_humor(fk_result['wbpos'].clone(), model, cc_cfg.robot_cfg.get("model", "smpl"))[:, :66]
    trans_vel, joints_humor_vel, root_orient_vel = estimate_velocities(torch.from_numpy(trans[None]), root_orient[None],
                                                                       joints_humor[None], 30)
    trans_vel = trans_vel[0]
    joints_humor_vel = joints_humor_vel[0]
    root_orient_vel = root_orient_vel[0]
    root_orient_mat = angle_axis_to_rotation_matrix(root_orient)[:, :3, :3]
    pose_body = pose_aa[:, 3:].reshape(-1, 23, 3)[:, :21]
    pose_body = angle_axis_to_rotation_matrix(pose_body.reshape(-1, 3))
    pose_body = pose_body.reshape(-1, 21, 4, 4)[:, :, :3, :3]

    #########################
    entry = {
        # "expert": fk_result,
        "pose_aa": pose_aa, # (B, 72)
        "pose_6d": pose_seq_6d, # (B, 144)
        "pose_body": pose_body, # (B, 21, 3, 3)
        "trans": trans, # (B, 3)
        "trans_vel": trans_vel, # (B, 3)
        "root_orient": root_orient_mat, # (B, 3, 3)
        "root_orient_vel": root_orient_vel, # (B, 3)
        "joints": joints_humor, # (B, 66)
        "joints_vel": joints_humor_vel, # (B, 66)
        "betas": mean_shape, # (16,)
        "seq_name": seq_name,
        "gender": gender,
        "joints2d": joints2d, # (B, 25, 3)
        "points3d": points3d, # (B, 25, 3)
        "cam": {
            "full_R": R,  # (3, 3)
            "full_t": tr,  # (3,)
            "K": K,
            "img_w": 900,
            "img_h": 900,
            "scene_name": seq_name[:-9]
        }
    }
    return dict_to_numpy(entry)