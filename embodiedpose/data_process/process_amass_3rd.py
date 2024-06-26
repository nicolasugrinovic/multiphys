import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse
import os
import sys
import time
import pickle
import glob
import pdb
import os.path as osp

sys.path.append(os.getcwd())

from tqdm import tqdm
import joblib
import yaml
from collections import defaultdict

from embodiedpose.utils import *
from embodiedpose.envs.humanoid_v2 import HumanoidEnv
from embodiedpose.data_loaders.statereg_dataset import Dataset
from embodiedpose.utils.arnet_3_config import Config
from embodiedpose.utils import get_qvel_fd, de_heading
from uhc.utils.numpy_smpl_humanoid import Humanoid
from embodiedpose.utils.transformation import quaternion_multiply, quaternion_inverse, rotation_from_quaternion
from embodiedpose.utils.transform_utils import (convert_6d_to_mat, compute_orth6d_from_rotation_matrix, convert_quat_to_6d)


def get_head_vel(head_pose, dt=1 / 30):
    # get head velocity
    head_vels = []
    head_qpos = head_pose[0]

    for i in range(head_pose.shape[0] - 1):
        curr_qpos = head_pose[i, :]
        next_qpos = head_pose[i + 1, :]
        v = (next_qpos[:3] - curr_qpos[:3]) / dt
        v = transform_vec(v, curr_qpos[3:7], 'heading')  # get velocity within the body frame

        qrel = quaternion_multiply(next_qpos[3:7], quaternion_inverse(curr_qpos[3:7]))
        axis, angle = rotation_from_quaternion(qrel, True)

        if angle > np.pi:  # -180 < angle < 180
            angle -= 2 * np.pi  #
        elif angle < -np.pi:
            angle += 2 * np.pi

        rv = (axis * angle) / dt
        rv = transform_vec(rv, curr_qpos[3:7], 'root')

        head_vels.append(np.concatenate((v, rv)))

    head_vels.append(head_vels[-1].copy())  # copy last one since there will be one less through finite difference
    head_vels = np.vstack(head_vels)
    return head_vels


def get_root_relative_head(root_poses, head_poses):
    res_root_poses = []

    for idx in range(head_poses.shape[0]):
        head_qpos = head_poses[idx]
        root_qpos = root_poses[idx]

        head_pos = head_qpos[:3]
        head_rot = head_qpos[3:7]
        q_heading = get_heading_q(head_rot).copy()
        obs = []

        root_pos = root_qpos[:3].copy()
        diff = root_pos - head_pos
        diff_loc = transform_vec(diff, head_rot, "heading")

        root_quat = root_qpos[3:7].copy()
        root_quat_local = quaternion_multiply(quaternion_inverse(head_rot), root_quat)  # ???? Should it be flipped?
        axis, angle = rotation_from_quaternion(root_quat_local, True)

        if angle > np.pi:  # -180 < angle < 180
            angle -= 2 * np.pi  #
        elif angle < -np.pi:
            angle += 2 * np.pi

        rv = axis * angle
        rv = transform_vec(rv, head_rot, 'root')  # root 2 head diff in head's frame

        root_pose = np.concatenate((diff_loc, rv))
        res_root_poses.append(root_pose)

    res_root_poses = np.array(res_root_poses)
    return res_root_poses


def root_from_relative_head(root_relative, head_poses):
    assert (root_relative.shape[0] == head_poses.shape[0])
    root_poses = []
    for idx in range(root_relative.shape[0]):
        head_pos = head_poses[idx][:3]
        head_rot = head_poses[idx][3:7]
        q_heading = get_heading_q(head_rot).copy()

        root_pos_delta = root_relative[idx][:3]
        root_rot_delta = root_relative[idx][3:]

        root_pos = quat_mul_vec(q_heading, root_pos_delta) + head_pos
        root_rot_delta = quat_mul_vec(head_rot, root_rot_delta)
        root_rot = quaternion_multiply(head_rot, quat_from_expmap(root_rot_delta))
        root_pose = np.hstack([root_pos, root_rot])
        root_poses.append(root_pose)
    return np.array(root_poses)


def get_obj_relative_pose(obj_poses, ref_poses, num_objs=1):
    # get object pose relative to the head
    res_obj_poses = []

    for idx in range(ref_poses.shape[0]):
        ref_qpos = ref_poses[idx]
        obj_qpos = obj_poses[idx]

        ref_pos = ref_qpos[:3]
        ref_rot = ref_qpos[3:7]
        q_heading = get_heading_q(ref_rot).copy()
        obs = []
        for oidx in range(num_objs):
            obj_pos = obj_qpos[oidx * 7:oidx * 7 + 3].copy()
            diff = obj_pos - ref_pos
            diff_loc = transform_vec(diff, ref_rot, "heading")

            obj_quat = obj_qpos[oidx * 7 + 3:oidx * 7 + 7].copy()
            obj_quat_local = quaternion_multiply(quaternion_inverse(q_heading), obj_quat)
            obj_pose = np.concatenate((diff_loc, obj_quat_local))
            obs.append(obj_pose)

        res_obj_poses.append(np.concatenate(obs))

    res_obj_poses = np.array(res_obj_poses)
    return res_obj_poses


def post_process_expert(expert, obj_pose, num_objs=1):
    head_pose = expert['head_pose']
    orig_traj = expert['qpos']
    root_pose = orig_traj[:, :7]

    head_vels = get_head_vel(head_pose)
    obj_relative_head = get_obj_relative_pose(obj_pose, head_pose, num_objs=num_objs)
    obj_relative_root = get_obj_relative_pose(obj_pose, root_pose, num_objs=num_objs)
    root_relative_2_head = get_root_relative_head(root_pose, head_pose)
    expert['head_vels'] = head_vels
    expert['obj_head_relative_poses'] = obj_relative_head
    expert['obj_root_relative_poses'] = obj_relative_root
    # root_recover = root_from_relative_head(root_relative_2_head, head_pose)
    # print(np.abs(np.sum(root_recover - root_pose)))
    return expert


def string_to_one_hot(all_classes, class_name):
    index = all_classes.index(class_name)
    hot = np.zeros(len(all_classes))
    hot[index] = 1
    return hot[None,]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='arnet_1')
    parser.add_argument('--render', action='store_true', default=False)
    args = parser.parse_args()

    # data_load = joblib.load("/hdd/zen/data/video_pose/h36m/data_fit/h36m_train_60_fitted_grad_qpos_full.p")
    data_load = joblib.load("/hdd/zen/data/video_pose/h36m/data_fit/h36m_train_30_fitted_grad_qpos_full.p")
    cfg = Config(args.cfg, create_dirs=False)

    of_folder = os.path.join(cfg.data_dir, 'fpv_of')

    num_sample = 0

    expert_dict = defaultdict(dict)
    pbar = tqdm((data_load.keys()))
    for take in pbar:
        v = data_load[take]
        expert = v['expert']

        pbar.set_description(take)
        # print(take, expert['len'], expert['qvel'].min(), expert['qvel'].max(), expert['head_height_lb'])
        expert['action'] = "all"
        if "obj_pose" in v:
            expert['obj_pose'] = v['obj_pose']
            expert['action_one_hot'] = np.repeat(np.array(string_to_one_hot(cfg.all_actions, action)), num_frames, axis=0)
            expert = post_process_expert(expert, v['obj_pose'], int(v['obj_pose'].shape[1] / 7))
        else:
            seq_len = expert['qpos'].shape[0]
            expert['obj_pose'] = np.repeat(np.array([[0, 0, 0, 1, 0, 0, 0]]), seq_len, axis=0)
            expert = post_process_expert(expert, expert['obj_pose'], int(expert['obj_pose'].shape[1] / 7))

        num_frames = expert['qpos'].shape[0]
        expert_dict[take] = expert

    # path = osp.join(cfg.data_dir, f"h36m_train_60_expert.p" )
    path = osp.join(cfg.data_dir, f"h36m_train_30_expert.p")
    print(f"{path}")
    joblib.dump(expert_dict, open(path, 'wb'))

    # path = osp.join(cfg.data_dir, "features", f"traj_all.p" )
    # print(f"{path}")
    # joblib.dump(expert_dict, open(path, 'wb'))
