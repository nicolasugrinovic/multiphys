'''
File: /humanoid_kin_v1.py
Created Date: Tuesday June 22nd 2021
Author: Zhengyi Luo
Comment:
-----
Last Modified: Tuesday June 22nd 2021 5:33:25 pm
Modified By: Zhengyi Luo at <zluo2@cs.cmu.edu>
-----
Copyright (c) 2022 Carnegie Mellon University, KLab
-----
'''

"""
This is the env used for the eval_scene, it seems that here is where everything happens: the simulation and the forward step
"""

from cmath import inf
from multiprocessing.spawn import get_preparation_data
from turtle import heading
import joblib
from numpy import isin
from scipy.linalg import cho_solve, cho_factor
import time
import pickle
from mujoco_py import functions as mjf
import mujoco_py
from gym import spaces
import os
import sys
import os.path as osp

sys.path.append(os.getcwd())

from uhc.khrylib.rl.envs.common import mujoco_env
from uhc.khrylib.utils import *
from uhc.khrylib.utils.transformation import quaternion_from_euler, quaternion_from_euler_batch
from uhc.khrylib.rl.core.policy_gaussian import PolicyGaussian
from uhc.khrylib.rl.core.critic import Value
from uhc.khrylib.models.mlp import MLP
from uhc.models.policy_mcp import PolicyMCP
from uhc.utils.flags import flags
from uhc.envs.humanoid_im import HumanoidEnv

from gym import spaces
from mujoco_py import functions as mjf
import pickle
import time
from scipy.linalg import cho_solve, cho_factor
import joblib
import numpy as np
import matplotlib.pyplot as plt

from uhc.smpllib.numpy_smpl_humanoid import Humanoid
# from uhc.smpllib.smpl_robot import Robot

from uhc.smpllib.smpl_mujoco import smpl_6d_to_qpose, smpl_to_qpose, qpos_to_smpl, smpl_to_qpose_torch
from uhc.utils.torch_geometry_transforms import (angle_axis_to_rotation_matrix as aa2mat, rotation_matrix_to_angle_axis as mat2aa)
import json
import copy

from embodiedpose.models.humor.utils.humor_mujoco import reorder_joints_to_humor, MUJOCO_2_SMPL
from embodiedpose.models.humor.humor_model import HumorModel
from embodiedpose.models.humor.utils.torch import load_state as load_humor_state
from embodiedpose.models.humor.body_model.utils import smpl_to_openpose
from embodiedpose.smpllib.scene_robot import SceneRobot
from embodiedpose.models.humor.utils.velocities import estimate_velocities
from embodiedpose.models.uhm_model import UHMModel
from scipy.spatial.transform import Rotation as sRot
import uhc.utils.pytorch3d_transforms as tR
from uhc.utils.tools import CustomUnpickler
import autograd.numpy as anp
from autograd import elementwise_grad as egrad

from autograd.misc import const_graph

from uhc.smpllib.np_smpl_humanoid_batch import Humanoid_Batch
import collections
from uhc.utils.math_utils import normalize_screen_coordinates, op_to_root_orient, smpl_op_to_op
from uhc.utils.torch_ext import isNpArray
from uhc.smpllib.smpl_parser import (
    SMPL_EE_NAMES,
    SMPL_BONE_ORDER_NAMES,
    SMPLH_BONE_ORDER_NAMES,
)

from utils.misc import save_img
import datetime
# ct stores current time
ct = datetime.datetime.now()
data_time = ct.strftime("%Y-%m-%d_%H-%M-%S")

def show_voxel(voxel_feat, name=None):
    num_grid = int(np.cbrt(voxel_feat.shape[0]))
    voxel_feat = voxel_feat.reshape(num_grid, num_grid, num_grid)
    x, y, z = np.indices((num_grid, num_grid, num_grid))
    colors = np.empty(voxel_feat.shape, dtype=object)
    colors[voxel_feat] = 'red'
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(voxel_feat, facecolors=colors, edgecolor='k')
    ax.view_init(16, 75)
    if name is None:
        plt.show()
    else:
        plt.savefig(name)
    plt.close()


class HumanoidKinEnvRes(HumanoidEnv):
    # Wrapper class that wraps around Copycat agent from UHC

    def __init__(self, kin_cfg, init_context, cc_iter=-1, mode="train", agent=None, num_agent=1, smpl_robot2=None):
        if smpl_robot2 is not None:
            self.smpl_robot2 = smpl_robot2
        self.num_agent = num_agent

        self.cc_cfg = cc_cfg = kin_cfg.cc_cfg
        self.kin_cfg = kin_cfg
        self.target = {}
        self.prev_humor_state = {}
        self.cur_humor_state = {}
        self.is_root_obs = None
        self.agent = agent
        # self.simulate = False
        self.simulate = True
        self.voxel_thresh = 0.1
        self.next_frame_idx = 250
        self.op_thresh = 0.1
        # self.n_ransac = 100

        self.load_context_pass = 0
        self.pred_joints2d = []

        # env specific
        self.use_quat = cc_cfg.robot_cfg.get("ball", False)
        cc_cfg.robot_cfg['span'] = kin_cfg.model_specs.get("voxel_span", 1.8)

        #### inside this object it this gets the scene! not at init though
        # masterfoot default is False, what does it do?
        # what is the difference btw smpl_robot and smpl_robot_orig?
        # this is used in HumanoidEnv(mujoco_env.MujocoEnv), located at uhc/envs/humanoid_im.py
        self.smpl_robot_orig = SceneRobot(cc_cfg.robot_cfg, data_dir=osp.join(cc_cfg.base_dir, "data/smpl"))

        self.hb = Humanoid_Batch(data_dir=osp.join(cc_cfg.base_dir, "data/smpl"))

        ############################# Agent loaded here #########################################
        # the important function is located in Robot class --> self.load_from_skeleton() where
        # SkeletonMesh() is instantiated
        # what is masterfoot?
        self.smpl_robot = SceneRobot(
            cc_cfg.robot_cfg,
            data_dir=osp.join(cc_cfg.base_dir, "data/smpl"),
            masterfoot=cc_cfg.masterfoot,
            num_agent=num_agent
        )
        # todo, dont add the second agent here, do it inside smpl_robot, just like add_simple_scene
        # but NOTE: the second robot has to be added before the simu is started, right?
        # maybe not necessarily becuase the scene can also be added afterwards
        #
        # self.smpl_robot2 = SceneRobot(
        #     cc_cfg.robot_cfg,
        #     data_dir=osp.join(cc_cfg.base_dir, "data/smpl"),
        #     masterfoot=cc_cfg.masterfoot,
        #     num_agent=2
        # )
        ##########################################################################################

        # this xml_str specifies only the human robot, lightning and cameras, and maybe the floor also
        # here each part of the body mesh for the simu is defined following the kinematic tree
        # also actuators are assigned to each joint.
        # probably I have to append the additional agent/robot in this xml_str
        self.xml_str = self.smpl_robot.export_xml_string().decode("utf-8")
        # self.xml_str2 = self.smpl_robot2.export_xml_string().decode("utf-8")
        if 0:
            fname = f"inspect_out/xml_s/robot1_{self.num_agent}.xml"
            self.smpl_robot.write_xml(fname)
            # fname = "inspect_out/xml_s/robot2.xml"
            # self.smpl_robot2.write_xml(fname)
        # in this XML:
        # the robot it defined in the body named <body name="Pelvis">, which is the root of the kinematic tree
        # and all the other joints or body parts are defined as children of this body
        # so to add a second robot, I have to add a second body with the same structure,
        # called for example <body name="Pelvis2">
        # NOTE: I prob. can add this just as add_simple_scene() adds to this file:
        # self.tree
        # self.smpl_robot.tree

        # print(self.xml_str )
        # print(self.xml_str2 )

        ''' Load Humor Model '''
        self.motion_prior = UHMModel(in_rot_rep="mat", out_rot_rep=kin_cfg.model_specs.get("out_rot_rep", "aa"),
                                     latent_size=24, model_data_config="smpl+joints", steps_in=1, use_gn=False)

        if self.kin_cfg.model_specs.get("use_rvel", False):
            self.motion_prior.data_names.append("root_orient_vel")
            self.motion_prior.input_dim_list += [3]

        if self.kin_cfg.model_specs.get("use_bvel", False):
            self.motion_prior.data_names.append("joints_vel")
            self.motion_prior.input_dim_list += [66]

        for param in self.motion_prior.parameters():
            param.requires_grad = False

        self.agg_data_names = self.motion_prior.data_names + ['points3d', "joints2d", "wbpos_cam", "beta"]

        if self.kin_cfg.model_specs.get("use_tcn", False):
            tcn_arch = self.kin_cfg.model_specs.get("tcn_arch", "3,3,3")
            filter_widths = [int(x) for x in tcn_arch.split(',')]
            self.num_context = int(np.prod(filter_widths))
            self.j2d_seq_feat = collections.deque([0] * self.num_context, self.num_context)

        self.body_grad = np.zeros(63)
        self.bm = bm = self.motion_prior.bm_dict['neutral']
        self.smpl2op_map = smpl_to_openpose(bm.model_type, use_hands=False, use_face=False, use_face_contour=False,
                                            openpose_format='coco25')
        self.smpl_2op_submap = self.smpl2op_map[self.smpl2op_map < 22]

        # if cfg.masterfoot:
        #     mujoco_env.MujocoEnv.__init__(self, cfg.mujoco_model_file)
        # else:
        #     mujoco_env.MujocoEnv.__init__(self, self.xml_str, 15)
        # print(self.xml_str)

        # the mujoco env is initialized here with the xml_str
        # is this not saved somewhere in a class property?
        # mujoco_env is an instance of the MujocoEnv class? but it is not saved in self
        # it loads a mujoco model at init() by  self.model = mujoco_py.load_model_from_xml(mujoco_model)
        # when doing mu_env_obj = mujoco_env, it returns None
        # before this, self.data is not defined

        mujoco_env.MujocoEnv.__init__(self, self.xml_str, 15)

        self.prev_qpos = self.data.qpos.copy() # (76, ), qpos is one per body in the xml, but
        # self.data.ctrl is (69,) and it is one per actuator
        # self.model.actuator_names
        # self.model.body_names # is tuple of (25)
        # self.model.joint_names # is tuple of (70)
        # self.model.nq # is 76
        # self.model._body_name2id # is 25, maybe it is 25*3=75 because it is 3 per body, but then why is nq=76? and not 75?

        self.setup_constants(cc_cfg, cc_cfg.data_specs, mode=mode, no_root=False)
        self.neutral_path = self.kin_cfg.data_specs['neutral_path']
        self.neutral_data = joblib.load(self.neutral_path)
        ###############################
        # this calls self.reset_robot() and SceneRobot.load_from_skeleton() is called from there
        if 0:
            fname = "inspect_out/xml_s/robot2_HumanoidKinEnvRes_init.xml"
            self.smpl_robot2.write_xml(fname)

        # this contains reset_robot(), and in turn it contains load_from_skeleton() and now self.smpl_robot.add_agent()
        # calling this will add the second agent, it this the best way to do it?
        self.load_context(init_context)
        ###############################
        # seems important: defines dimensions of the action space and thus of the policy net
        # I think this should be done per agent
        self.set_action_spaces()
        #  MPG is called inside this function by calling get_humor_dict_obs_from_sim
        self.set_obs_spaces()
        self.weight = mujoco_py.functions.mj_getTotalmass(self.model)

        ''' Load CC Controller '''
        self.state_dim = state_dim = self.get_cc_obs().shape[0]
        cc_action_dim = self.action_dim
        # selects cc_policy
        if cc_cfg.actor_type == "gauss":
            self.cc_policy = PolicyGaussian(cc_cfg, action_dim=cc_action_dim, state_dim=state_dim)
        elif cc_cfg.actor_type == "mcp":
            self.cc_policy = PolicyMCP(cc_cfg, action_dim=cc_action_dim, state_dim=state_dim)

        self.cc_value_net = Value(MLP(state_dim, cc_cfg.value_hsize, cc_cfg.value_htype))
        if cc_iter != -1:
            cp_path = '%s/iter_%04d.p' % (cc_cfg.model_dir, cc_iter)
        else:
            cc_iter = np.max([int(i.split("_")[-1].split(".")[0]) for i in os.listdir(cc_cfg.model_dir)])
            cp_path = '%s/iter_%04d.p' % (cc_cfg.model_dir, cc_iter)

        print((f'loading model from checkpoint in {__name__}: %s' % cp_path))
        model_cp = CustomUnpickler(open(cp_path, "rb")).load()
        self.cc_running_state = model_cp['running_state']
        # loads checkpoint
        self.cc_policy.load_state_dict(model_cp['policy_dict'])
        self.cc_value_net.load_state_dict(model_cp['value_dict'])

        # Contact modelling
        body_id_list = self.model.geom_bodyid.tolist()
        self.contact_geoms = [body_id_list.index(self.model._body_name2id[body]) for body in SMPL_BONE_ORDER_NAMES]

    # def set_smpl_robot2(self, smpl_robot):
    #     self.smpl_robot2 = smpl_robot

    def reset_robot(self):
        beta = self.context_dict["beta"].copy()
        gender = self.context_dict["gender"].copy()
        # this seems important for loading the scene, for chi3d this is 's'-->why?
        scene_name = self.context_dict['cam']['scene_name']

        if "obj_info" in self.context_dict:
            obj_info = self.context_dict['obj_info']
            self.smpl_robot.load_from_skeleton(torch.from_numpy(beta[0:1, :]).float(), gender=gender, obj_info=obj_info)
        else:
            # this seems important for loading the scene
            if not self.context_dict.get("load_scene", True):
                scene_name = None
            ####################################################################################################
            # Scene loading with the function add_simple_scene is done here
            # loads humanoid and simulation environment from template file and modifies it according to beta and gender
            self.smpl_robot.load_from_skeleton(torch.from_numpy(beta[0:1, :]).float(), gender=gender,
                                               scene_and_key=scene_name, num_agent=self.num_agent)
            # first define the agent2 and then when agent 1 is loading do add_agent with agent2.tree as arg
            # if self.kin_cfg.two_agents:
            if hasattr(self, 'smpl_robot2'):
                print('**** Operating with 2 AGENTS ****')
                # todo add second agent from to self.smpl_robot. here
                # here exists self.smpl_robot2,
                # in the first pass when called from agent = agent_class() this works OK, but in the second pass when
                # called by SceneVisulizer() it does not work
                # self.smpl_robot.add_agent(self.smpl_robot2.tree, agent_id=2)
                self.smpl_robot.add_agent_v2(self.smpl_robot2, agent_id=2)
            ####################################################################################################

        xml_str = self.smpl_robot.export_xml_string().decode("utf-8")
        if 0:
            # from utils.misc import write_txt
            fname = "inspect_out/xml_s/humanoidKinEnvRes_reset_robot_add_agent_after.xml"
            # write_txt(fname, xml_str)
            self.smpl_robot.tree.write(fname, pretty_print=True)
        ######################################
        # reloads the simulation using mujoco_py.load_model_from_xml(xml_str) and deletes de old viewers,
        # gets init_qpos and init_qvel
        self.reload_sim_model(xml_str)
        ######################################
        self.weight = self.smpl_robot.weight
        # hb is a Humanoid_Batch instance, it contains proj_2d_loss, the np version not the torch one found in the same script
        self.hb.update_model(torch.from_numpy(beta[0:1, :16]), torch.tensor(gender[0:1]))
        self.hb.update_projection(self.camera_params, self.smpl2op_map, MUJOCO_2_SMPL)
        self.proj_2d_loss = egrad(self.hb.proj_2d_loss) # egrad is elementwise_grad from autograd
        self.proj_2d_body_loss = egrad(self.hb.proj_2d_body_loss)
        self.proj_2d_root_loss = egrad(self.hb.proj_2d_root_loss)
        self.proj_2d_line_loss = egrad(self.hb.proj_2d_line_loss)
        return xml_str # this return is ingnored

    def load_context(self, data_dict):
        # the simulation FOR LOOP starts when load_context_pass=2
        self.load_context_pass += 1
        self.context_dict = {k: v.squeeze().cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in data_dict.items()}

        self.camera_params = data_dict['cam']
        self.camera_params_torch = {k: torch.from_numpy(v).double() if isNpArray(v) else v for k, v in self.camera_params.items()}

        ######################################
        # this does a lot!
        # it reloads the simulation using mujoco_py.load_model_from_xml(xml_str) based on body betas
        # defines the reprojection losses self.proj_2d_loss and adds gradient computations to them
        # this calls self.smpl_robot.load_from_skeleton
        self.reset_robot()
        ######################################
        # this next function update_model() handles var self.body_name
        self.humanoid.update_model(self.model)
        ######################################

        self.context_dict['len'] = self.context_dict['pose_aa'].shape[0] - 1
        gt_qpos = smpl_to_qpose(self.context_dict['pose_aa'], self.model, trans=self.context_dict['trans'], count_offset=True)

        # this is the HUMOR estimated pose
        if 0:
            from utils.body_model import pose_to_vertices as pose_to_vertices_
            import smplx
            import trimesh
            from functools import partial
            from utils.misc import save_trimesh
            local_bm = smplx.create("data", 'smpl', use_pca=False, batch_size=1).cuda()
            pose_to_vertices = partial(pose_to_vertices_, pose_type="smpl", alpha=1, bm=local_bm)
            def smpl_to_verts(humor_first_pose, init_trans):
                pose = np.concatenate([humor_first_pose, init_trans], axis=1)
                pose = torch.from_numpy(pose).float().cuda()
                verts = pose_to_vertices(pose[None])
                return verts

            inspect_path = f"inspect_out/h_kinres/chi3d/{data_time}/{self.load_context_pass}/"
            humor_first_pose = self.context_dict['init_pose_aa'][None,]
            init_trans = self.context_dict['init_trans'][None,]
            pose = np.concatenate([humor_first_pose, init_trans], axis=1)
            pose = torch.from_numpy(pose).float().cuda()
            verts = pose_to_vertices(pose[None])
            save_trimesh(verts[0,0], local_bm.faces, inspect_path+"humor_first_pose.ply")

            # no trans
            no_trans = np.zeros_like(self.context_dict['init_trans'][None,])
            pose = np.concatenate([humor_first_pose, no_trans], axis=1)
            pose = torch.from_numpy(pose).float().cuda()
            verts = pose_to_vertices(pose[None])
            save_trimesh(verts[0,0], local_bm.faces, inspect_path+"humor_first_pose_no_trans.ply")

            # verts bbox
            mesh = trimesh.Trimesh(vertices=verts[0,0].detach().cpu().numpy(), faces=local_bm.faces)
            bbox = mesh.bounding_box.bounds
            min_xyz = bbox[0]
            to_floor_trans = init_trans - np.array([[0., 0., min_xyz[2]]])
            verts_floor = smpl_to_verts(humor_first_pose, to_floor_trans)
            save_trimesh(verts_floor[0,0], local_bm.faces, inspect_path+"humor_first_pose_floor.ply")

            # floor no trans
            no_trans = np.zeros_like(self.context_dict['init_trans'][None,])
            to_floor_no_trans = no_trans - np.array([[0., 0., min_xyz[2]]])
            pose = np.concatenate([humor_first_pose, to_floor_no_trans], axis=1)
            pose = torch.from_numpy(pose).float().cuda()
            verts = pose_to_vertices(pose[None])
            save_trimesh(verts[0,0], local_bm.faces, inspect_path+"humor_first_pose_floor_no_trans.ply")

        # something very hacky to correct the initial translation
        if 0:
            # to_floor_trans = np.array([[-0.581167008413573, -0.6478122600077725, 0.8168622255325317]])
            # self.context_dict['init_trans'] = to_floor_trans[0]
            self.context_dict['init_trans'] = to_floor_no_trans[0]

        # this is the HUMOR estimated pose
        init_qpos = smpl_to_qpose(self.context_dict['init_pose_aa'][None,], self.model,
                                  trans=self.context_dict['init_trans'][None,],
                                  count_offset=True)
        self.context_dict["qpos"] = gt_qpos

        # uses as input  the initial HUMOR estimate, first pose, it serves as first target pose
        self.target = self.humanoid.qpos_fk(torch.from_numpy(init_qpos))
        # contains the keys (['trans', 'root_orient', 'pose_body', 'joints', 'root_orient_vel', 'joints_vel'])
        self.prev_humor_state = {k: data_dict[k][:, 0:1, :].clone() for k in self.motion_prior.data_names}
        self.cur_humor_state = self.prev_humor_state
        #####
        # self.humanoid located at torch_smpl_humanoid.py
        # self.gt_targets --> keys are (['qpos', 'qvel', 'wbpos', 'wbquat', 'bquat', 'body_com', 'rlinv', 'rlinv_local',
        # 'rangv', 'bangvel', 'ee_wpos', 'ee_pos', 'com', 'height_lb', 'len'])
        self.gt_targets = self.humanoid.qpos_fk(torch.from_numpy(gt_qpos))

        self.target.update({k: data_dict[k][:, 0:1, :].clone() for k in self.motion_prior.data_names})  # Initializing target

        if self.kin_cfg.model_specs.get("use_tcn", False):
            # this is the HUMOR pose to convert it to world coordinates
            world_body_pos = self.target['wbpos'].reshape(24, 3)[MUJOCO_2_SMPL][self.smpl_2op_submap]
            if 0:
                from utils.misc import save_pointcloud
                inspect_path = f"inspect_out/h_kinres/chi3d_rot/{data_time}/{self.load_context_pass}/"
                save_pointcloud(world_body_pos, inspect_path + f"world_body_pos.ply")
                save_pointcloud(self.target['wbpos'].reshape(24, 3), inspect_path + f"target_wbpos.ply")

            world_trans = world_body_pos[..., 7:8:, :]
            self.pred_tcn = {
                'world_body_pos': world_body_pos - world_trans,
                'world_trans': world_trans,
            }

            casual = self.kin_cfg.model_specs.get("casual_tcn", True)
            full_R, full_t = self.camera_params["full_R"], self.camera_params['full_t']
            if casual: # in the multiphys specs casual_tcn: true
                joints2d = self.context_dict["joints2d"][0:1].copy() # shape (1, 12, 3)
                joints2d[joints2d[..., 2] < self.op_thresh] = 0 # op_thresh=0.1
                # normalizes joints2d from screen coordinates to unit coordinates
                # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
                joints2d[..., :2] = normalize_screen_coordinates(joints2d[..., :2], self.camera_params['img_w'],
                                                                 self.camera_params['img_h'])
                joints2d = np.pad(joints2d, ((self.num_context - 1, 0), (0, 0), (0, 0)), mode="edge")
            else:
                joints2d = self.context_dict["joints2d"][:(self.num_context // 2 + 1)].copy()
                joints2d[joints2d[..., 2] < self.op_thresh] = 0
                joints2d[..., :2] = normalize_screen_coordinates(joints2d[..., :2], self.camera_params['img_w'],
                                                                 self.camera_params['img_h'])
                joints2d = np.pad(joints2d, ((self.num_context // 2, self.num_context // 2 + 1 - joints2d.shape[0]),
                                             (0, 0), (0, 0)), mode="edge")

            # it enters the else
            if self.kin_cfg.model_specs.get("tcn_3dpos", False):
                world_body_pos = self.target['wbpos'].reshape(24, 3)[MUJOCO_2_SMPL][self.smpl_2op_submap]
                world_body_pos = smpl_op_to_op(world_body_pos)
                cam_body_pos = world_body_pos @ full_R.T + full_t
                j2d3dfeat = np.concatenate([joints2d[..., :2], np.repeat(cam_body_pos[None,],
                                                                         self.num_context, axis=0)], axis=-1)

                [self.j2d_seq_feat.append(j3dfeat) for j3dfeat in j2d3dfeat]
                self.pred_tcn['cam_body_pos'] = cam_body_pos
            else:
                # at this point the j2dfeat is the normalized 2d joints
                [self.j2d_seq_feat.append(j2dfeat) for j2dfeat in joints2d[..., :2]]


    def set_model_params(self):
        if self.cc_cfg.action_type == 'torque' and hasattr(self.cc_cfg, 'j_stiff'):
            self.model.jnt_stiffness[1:] = self.cc_cfg.j_stiff
            self.model.dof_damping[6:] = self.cc_cfg.j_damp

    def get_obs(self):
        ar_obs = self.get_ar_obs_v1()
        return ar_obs

    def get_cc_obs(self):
        # it enters uhc/envs/humanoid_im.py and uses the function self.get_full_obs_v2()
        return super().get_obs()

    def get_ar_obs_v1(self):
        t = self.cur_t
        obs = []
        compute_root_obs = False
        if self.is_root_obs is None:
            self.is_root_obs = []
            compute_root_obs = True

        curr_qpos = self.data.qpos[:self.qpos_lim].copy()
        curr_qvel = self.data.qvel[:self.qvel_lim].copy()
        self.prev_humor_state = copy.deepcopy(self.cur_humor_state)

        #################### MPG is called inside this function ##############
        # MPG is called inside this function
        # proj2dgrad comes from this function
        self.cur_humor_state = humor_dict = self.get_humor_dict_obs_from_sim()
        self.pred_joints2d.append(humor_dict['pred_joints2d'])
        ######################################################################

        curr_root_quat = self.remove_base_rot(curr_qpos[3:7])
        full_R, full_t = self.camera_params_torch['full_R'], self.camera_params_torch['full_t']
        target_global_dict = {k: torch.from_numpy(self.context_dict[k][(t + 1):(t + 2)].reshape(humor_dict[k].shape)) for k in self.motion_prior.data_names}
        conon_output = self.motion_prior.canonicalize_input_double(humor_dict, target_global_dict,
                                                                   split_input=False, return_info=True)
        humor_local_dict, next_target_local_dict, info_dict = conon_output
        # print(torch.matmul(humor_dict['trans'], full_R.T) + full_t)
        # info_dict --> keys (['world2aligned_trans', 'world2aligned_rot', 'trans2joint'])
        heading_rot = info_dict['world2aligned_rot'].numpy()
        body_obs_list = [humor_local_dict[k].flatten().numpy() for k in self.motion_prior.data_names]
        body_obs_list[2].reshape([-1, 3]).shape
        curr_body_obs = np.concatenate(body_obs_list)

        # hq = get_heading_new(curr_qpos[3:7])
        hq = 0
        obs.append(np.array([hq])) # (1,)
        obs.append(curr_body_obs) # (336,)
        if compute_root_obs:
            self.is_root_obs.append(np.array([1]))
            self.is_root_obs.append(np.concatenate([[1 if "root" in k else 0] * humor_local_dict[k].flatten().numpy().shape[-1] for k in self.motion_prior.data_names]))

        if self.kin_cfg.model_specs.get("use_tcn", False):
            casual = self.kin_cfg.model_specs.get("casual_tcn", True)
            if casual:
                joints2d_gt = self.context_dict['joints2d'][self.cur_t + 1].copy()
                joints2d_gt[..., :2] = normalize_screen_coordinates(joints2d_gt[..., :2],
                                                                    self.camera_params['img_w'],
                                                                    self.camera_params['img_h'])
                joints2d_gt[joints2d_gt[..., 2] < self.op_thresh] = 0
            else:
                t = self.cur_t + 1
                pad_num = self.num_context // 2 + 1
                joints2d_gt = self.context_dict['joints2d'][t:(t + pad_num)].copy()
                if joints2d_gt.shape[0] < pad_num:
                    joints2d_gt = np.pad(joints2d_gt, ([0, pad_num - joints2d_gt.shape[0]], [0, 0], [0, 0]), mode="edge")

                joints2d_gt[..., :2] = normalize_screen_coordinates(joints2d_gt[..., :2],
                                                                    self.camera_params['img_w'],
                                                                    self.camera_params['img_h'])
                joints2d_gt[joints2d_gt[..., 2] < self.op_thresh] = 0

            if 0:
                from utils.misc import plot_joints_cv2
                black = np.zeros([1080, 1920, 3], dtype=np.uint8)
                # black = np.zeros([900, 900, 3], dtype=np.uint8)
                j2d_gt = self.context_dict['joints2d'][self.cur_t + 1].copy()
                plot_joints_cv2(black, j2d_gt[None], show=True, with_text=True, sc=3)

            if self.kin_cfg.model_specs.get("tcn_3dpos", False):
                # cam_pred_tcn_3d = humor_dict['cam_pred_tcn_3d']
                # j2d3dfeat = np.concatenate([joints2d_gt[..., :2], cam_pred_tcn_3d.numpy().squeeze()], axis = 1)
                cam_pred_3d = humor_dict['cam_pred_3d']
                cam_pred_3d = smpl_op_to_op(cam_pred_3d)
                if casual:
                    j2d3dfeat = np.concatenate([joints2d_gt[..., :2], cam_pred_3d.squeeze()], axis=1)
                    self.j2d_seq_feat.append(j2d3dfeat)  # push next step obs into state
                else:
                    j2d3dfeat = np.concatenate([joints2d_gt[..., :2], np.repeat(cam_pred_3d.squeeze(1), self.num_context // 2 + 1, axis=0)], axis=-1)
                    [self.j2d_seq_feat.pop() for _ in range(self.num_context // 2)]
                    [self.j2d_seq_feat.append(feat) for feat in j2d3dfeat]
            ########################## NUK: it enters here ############################################
            else:
                if casual: # what is j2d_seq_feat?
                    self.j2d_seq_feat.append(joints2d_gt[:, :2])  # push next step obs into state
                else:
                    [self.j2d_seq_feat.pop() for _ in range(self.num_context // 2)]
                    [self.j2d_seq_feat.append(feat) for feat in joints2d_gt[..., :2]]
            ###########################################################################################
            j2d_seq = np.array(self.j2d_seq_feat).flatten()
            obs.append(j2d_seq) # j2d_seq shape: (1944,) are 81 flattened 2d joints
            if compute_root_obs:
                self.is_root_obs.append(np.array([3] * j2d_seq.shape[0]))

            # use tcn directly on the projection gradient
            tcn_root_grad = self.kin_cfg.model_specs.get("tcn_root_grad", False)
            world_body_pos, world_trans = self.pred_tcn['world_body_pos'], self.pred_tcn['world_trans']
            curr_body_jts = humor_dict['joints'].reshape(22, 3)[self.smpl_2op_submap].numpy() # (14, 3)
            curr_body_jts -= curr_body_jts[..., 7:8, :] # root relative?
            world_body_pos -= world_body_pos[..., 7:8, :]
            body_diff = transform_vec_batch_new(world_body_pos - curr_body_jts, curr_root_quat).T.flatten()
            if 0:
                from utils.misc import save_pointcloud
                inspect_path = f"inspect_out/prox/get_ar_obs_v1/"
                save_pointcloud(world_body_pos, inspect_path + f"world_body_pos_{t:03d}.ply")
                save_pointcloud(curr_body_jts, inspect_path + f"curr_body_jts_{t:03d}.ply")

            if self.kin_cfg.model_specs.get("tcn_body", False):
                obs.append(body_diff)

            curr_trans = self.target['wbpos'][:, :3]  # this is in world coord
            trans_diff = np.matmul(world_trans - curr_trans, heading_rot[0].T).flatten()
            trans_diff[2] = world_trans[:, 2]  # Mimicking the target trans feat.
            if self.kin_cfg.model_specs.get("tcn_traj", False):
                obs.append(trans_diff)

            if not tcn_root_grad:
                pred_root_mat = op_to_root_orient(world_body_pos[None,])
                root_rot_diff = np.matmul(heading_rot, pred_root_mat).flatten()
                obs.append(root_rot_diff) # (9, )

            if self.kin_cfg.model_specs.get("tcn_body", False):
                if compute_root_obs:
                    self.is_root_obs.append(np.array([0] * body_diff.shape[0]))

            if self.kin_cfg.model_specs.get("tcn_traj", False):
                if compute_root_obs:
                    self.is_root_obs.append(np.array([1] * trans_diff.shape[0]))

            if not tcn_root_grad:
                if compute_root_obs:
                    self.is_root_obs.append(np.array([1] * root_rot_diff.shape[0]))

        if self.kin_cfg.model_specs.get("use_rt", True):
            trans_target_local = next_target_local_dict['trans'].flatten().numpy()
            obs.append(trans_target_local)
            if compute_root_obs:
                self.is_root_obs.append(np.array([1] * trans_target_local.shape[0]))

        if self.kin_cfg.model_specs.get("use_rr", False):
            root_rot_diff = next_target_local_dict['root_orient'].flatten().numpy()
            obs.append(root_rot_diff)
            if compute_root_obs:
                self.is_root_obs.append(np.array([1] * root_rot_diff.shape[0]))

        if self.kin_cfg.model_specs.get("use_3d_grad", False):
            normalize = self.kin_cfg.model_specs.get("normalize_3d_grad", True)
            proj2dgrad = humor_dict['proj2dgrad'].squeeze().numpy().copy()
            proj2dgrad = np.nan_to_num(proj2dgrad, nan=0, posinf=0, neginf=0)
            proj2dgrad = np.clip(proj2dgrad, -200, 200)

            if normalize:
                body_mul = root_mul = 1
            else:
                grad_mul = self.kin_cfg.model_specs.get("grad_mul", 10)
                body_mul = (10 * grad_mul)
                root_mul = (100 * grad_mul)

            trans_grad = (np.matmul(heading_rot, proj2dgrad[:3]) / root_mul).squeeze()
            root_grad = (sRot.from_matrix(heading_rot) * sRot.from_rotvec(proj2dgrad[3:6] / body_mul)).as_rotvec().squeeze()
            body_grad = proj2dgrad[6:69] / body_mul

            obs.append(trans_grad)
            if compute_root_obs:
                self.is_root_obs.append(np.array([1] * trans_grad.shape[0]))
            obs.append(root_grad)
            if compute_root_obs:
                self.is_root_obs.append(np.array([1] * root_grad.shape[0]))
            obs.append(body_grad)
            if compute_root_obs:
                self.is_root_obs.append(np.array([1] * body_grad.shape[0]))
        ##### it enters here ##########################################################################################
        elif self.kin_cfg.model_specs.get("use_3d_grad_adpt", False):
            no_grad_body = self.kin_cfg.model_specs.get("no_grad_body", False)
            proj2dgrad = humor_dict['proj2dgrad'].squeeze().numpy().copy()
            proj2dgrad = np.nan_to_num(proj2dgrad, nan=0, posinf=0, neginf=0)
            proj2dgrad = np.clip(proj2dgrad, -200, 200)
            # sRot if from scipy.spatial.transform rotation
            trans_grad = (np.matmul(heading_rot, proj2dgrad[:3])).squeeze()
            root_grad = (sRot.from_matrix(heading_rot) * sRot.from_rotvec(proj2dgrad[3:6])).as_rotvec().squeeze()
            body_grad = proj2dgrad[6:69]
            if no_grad_body:
                # Ablation, zero body grad. Just TCN
                body_grad = np.zeros_like(body_grad)
            obs.append(trans_grad) # (3,)
            if compute_root_obs:
                self.is_root_obs.append(np.array([1] * trans_grad.shape[0]))
            obs.append(root_grad) # (3,)
            if compute_root_obs:
                self.is_root_obs.append(np.array([1] * root_grad.shape[0]))
            obs.append(body_grad) # (63,)
            if compute_root_obs:
                self.is_root_obs.append(np.array([1] * body_grad.shape[0]))
        ################################################################################################################

        if self.kin_cfg.model_specs.get("use_sdf", False):
            sdf_vals = self.smpl_robot.get_sdf_np(self.cur_humor_state['joints'].reshape(-1, 3), topk=3)
            obs.append(sdf_vals.numpy().flatten())
            if compute_root_obs:
                self.is_root_obs.append(np.array([2] * sdf_vals.shape[0]))
        elif self.kin_cfg.model_specs.get("use_dir_sdf", False):
            sdf_vals, sdf_dirs = self.smpl_robot.get_sdf_np(self.cur_humor_state['joints'].reshape(-1, 3), topk=3, return_grad=True)
            sdf_dirs = np.matmul(sdf_dirs, heading_rot[0].T)  # needs to be local dir coord
            sdf_feat = (sdf_vals[:, :, None] * sdf_dirs).numpy().flatten()
            obs.append(sdf_feat)
            if compute_root_obs:
                self.is_root_obs.append(np.array([2] * sdf_feat.shape[0]))
        ################ VOXEL observations ############################################################################
        ########################### it enters here #####################################################################
        if self.kin_cfg.model_specs.get("use_voxel", False):
            voxel_res = self.kin_cfg.model_specs.get("voxel_res", 8) # this is =16
            # these voxel_feat are float continuous values
            voxel_feat = self.smpl_robot.query_voxel(self.cur_humor_state['trans'].reshape(-1, 3),
                                                     self.cur_humor_state['root_orient'].reshape(3, 3),
                                                     res=voxel_res).flatten() # (4096,)
            # these are booleans of shape (4096,) and self.voxel_thresh=0.1
            inside, outside = voxel_feat <= 0, voxel_feat >= self.voxel_thresh

            if 0:
                from skimage import measure
                from skimage.draw import ellipsoid
                import trimesh
                inside_ = inside.reshape(voxel_res, voxel_res, voxel_res)
                inside_np = inside_[:, :, 1]
                verts, faces, normals, values = measure.marching_cubes(inside_, 0.0)
                mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
                mesh.export('voxel_features.ply')

            middle = np.logical_and(~inside, ~outside)
            # voxel_feat has values different from 0 and 1 due to middle
            voxel_feat[inside], voxel_feat[outside] = 1, 0
            voxel_feat[middle] = (self.voxel_thresh - voxel_feat[middle]) / self.voxel_thresh
            # voxel_feat.min()
            # voxel_feat.max()
            # voxel_feat.sum()
            # voxel_feat[:] = 0
            if compute_root_obs:
                self.is_root_obs.append(np.array([2] * voxel_feat.shape[0]))
            obs.append(voxel_feat) #  (4096,)
        ################################################################################################################

        if self.kin_cfg.model_specs.get("use_contact", False):
            contact_feat = np.zeros(24)
            for contact in self.data.contact[:self.data.ncon]:
                g1, g2 = contact.geom1, contact.geom2
                if g1 in self.contact_geoms and not g2 in self.contact_geoms:
                    contact_feat[g1 - 1] = 1
                if g2 in self.contact_geoms and not g1 in self.contact_geoms:
                    contact_feat[g2 - 1] = 1
            if compute_root_obs:
                self.is_root_obs.append(np.array([0] * contact_feat.shape[0]))
            obs.append(contact_feat)

        # voxel_feat_show = self.smpl_robot.query_voxel(
        #     self.cur_humor_state['trans'].reshape(-1, 3),
        #     self.cur_humor_state['root_orient'].reshape(3, 3),
        #     res=16).flatten()
        # os.makedirs(osp.join("temp", self.context_dict['seq_name']), self.cfg.id, exist_ok=True)
        # show_voxel(voxel_feat_show <= 0.05, name = osp.join("temp", self.context_dict['seq_name'], self.cfg.id, f"voxel_{self.cur_t:05d}.png"))
        # show_voxel(voxel_feat_show <= 0.05, name = None)

        obs = np.concatenate(obs) # (6455,)
        if compute_root_obs:
            self.is_root_obs = np.concatenate(self.is_root_obs)
            assert (self.is_root_obs.shape == obs.shape)

        return obs

    def step_ar(self, action, dt=1 / 30):
        # action is (114,)
        cfg = self.kin_cfg

        # change this (action) to number of joints
        # cur_humor_state --> (['trans_vel', 'joints_vel', 'root_orient_vel', 'joints',
        # 'pose_body', 'root_orient', 'trans', 'pred_joints2d', 'cam_pred_3d', 'proj2dgrad'])
        action_ = torch.from_numpy(action[None, :69])

        #### motion_prior.step_state() ####
        # this actually ONLY does some transformations and
        # applies the predicted residuals from the kinematic policy to the pose
        next_global_out = self.motion_prior.step_state(self.cur_humor_state, action_)
        # next_global_out --> includes (['trans', 'root_orient', 'pose_body'])

        body_pose_aa = mat2aa(next_global_out['pose_body'].reshape(21, 3, 3)).reshape(1, 63)
        root_aa = mat2aa(next_global_out['root_orient'].reshape(1, 3, 3)).reshape(1, 3)
        pose_aa = torch.cat([root_aa, body_pose_aa, torch.zeros(1, 6).to(root_aa)], dim=1) # (1, 72)

        # get the GT data from self.context_dict, possibly 'pose_body'
        overwrite_target_w_gt = self.kin_cfg.overwrite_target_w_gt
        if overwrite_target_w_gt:
            if self.cur_t == 0:
                print("*** Warning: OVERRIDING TARGET WITH GT ***")
            gt_pose = torch.from_numpy(self.context_dict['pose_body'].reshape(-1, 3, 3))
            body_pose_aa_gt = mat2aa(gt_pose).reshape(-1, 63)
            N = body_pose_aa_gt.shape[0]
            gt_root = torch.from_numpy(self.context_dict['root_orient'].reshape(-1, 3, 3))
            root_aa_gt = mat2aa(gt_root).reshape(-1, 3)
            pose_aa_gt = torch.cat([root_aa_gt, body_pose_aa_gt, torch.zeros(N, 6).to(root_aa)], dim=1)  # (1, 72)
            trans_gt = torch.from_numpy(self.context_dict["trans"])
            # self.cur_t
            pose_aa = pose_aa_gt[self.cur_t].reshape(1, -1)
            next_global_out['trans'] = trans_gt[self.cur_t].reshape(1, 1, -1)
            next_global_out['root_orient'] = gt_root[self.cur_t].reshape(1, 1, -1)
            gt_pose_ = gt_pose.reshape(-1, 21, 3, 3).reshape(-1, 21 * 3 * 3)
            next_global_out['pose_body'] = gt_pose_[self.cur_t].reshape(1, 1, -1)


        qpos = smpl_to_qpose_torch(pose_aa, self.model, trans=next_global_out['trans'].reshape(1, 3),
                                   count_offset=True) # (1, 76)

        if self.mode == "train" and self.agent.iter < self.agent.num_supervised and self.agent.iter >= 0:
            # Dagger
            qpos = torch.from_numpy(self.gt_targets['qpos'][(self.cur_t):(self.cur_t + 1)])
            fk_res = self.humanoid.qpos_fk(qpos)
        else:
            # fk_res --> keys:
            # (['qpos', 'qvel', 'wbpos', 'wbquat', 'bquat', 'body_com', 'rlinv', 'rlinv_local',
            # 'rangv', 'bangvel', 'ee_wpos', 'ee_pos', 'com', 'height_lb', 'len'])
            fk_res = self.humanoid.qpos_fk(qpos)

        self.target = fk_res
        # updates the target with the next pose dictated by the kinematic motion prior (HUMOR)
        # all the quantities here are for one time step or one frame only
        self.target.update(next_global_out)
        if self.kin_cfg.model_specs.get("use_tcn", False):
            full_R, full_t = self.camera_params['full_R'], self.camera_params['full_t']
            kp_feats = action[69:].copy()
            cam_trans = kp_feats[None, :3]
            cam_body_pos = kp_feats[3:].reshape(14, 3)

            # camera to world transformation
            self.pred_tcn['world_trans'] = (cam_trans - full_t).dot(full_R)
            # world_body_pos are the (prev_step?) joints from pose_aa or next_global_out (which is the same)
            # it comes from action
            world_body_pos = cam_body_pos.dot(full_R)
            self.pred_tcn['world_body_pos'] = world_body_pos
            self.pred_tcn['cam_body_pos'] =  cam_trans + cam_body_pos

        # debug visu
        if 0:
            from utils.smpl import smpl_to_verts
            from utils.misc import save_trimesh
            from utils.misc import save_pointcloud
            t = self.cur_t
            trans = next_global_out["trans"]
            inspect_path = f"inspect_out/sim_loop/chi3d/{data_time}/{t:03d}/"
            next_prior_verts, faces = smpl_to_verts(pose_aa, trans[0])
            save_trimesh(next_prior_verts[0, 0], faces, inspect_path + f"next_prior_verts_{t:03d}.ply")
            save_pointcloud(world_body_pos, inspect_path + f"world_body_pos_{t:03d}.ply")
            pred_joints2d = self.cur_humor_state["pred_joints2d"]
            # this pred_joints2d has very weird values for CHI3D data
            pred_joints2d_ = pred_joints2d[0, 0].cpu().numpy()



    def get_humanoid_pose_aa_trans(self, qpos=None):
        if qpos is None:
            qpos = self.data.qpos.copy()[None]
        pose_aa, trans = qpos_to_smpl(qpos, self.model, self.cc_cfg.robot_cfg.get("model", "smpl"))

        return pose_aa, trans

    def get_humor_dict_obs_from_sim(self):
        """ NUK: Compute obs based on current and previous simulation state and coverts it into humor format. """
        # gets both the current and previous qpos
        # self.data is the mujoco data, so this comes from mujoco loaded simulation
        qpos = self.data.qpos.copy()[None] # (1, 76) but with agent2 qpos is (1, 152)
        # NUK hack for now
        qpos = qpos[:, :76]
        # qpos = self.get_expert_qpos()[None] # No simulate
        prev_qpos = self.prev_qpos[None] # (1, 76)
        # NUK hack for now
        prev_qpos = prev_qpos[:, :76]

        # Calculating the velocity difference from simulation. We do not use target velocity. 
        qpos_stack = np.concatenate([prev_qpos, qpos])
        pose_aa, trans = self.get_humanoid_pose_aa_trans(qpos_stack) # Simulation state.

        if 0:
            from utils.smpl import smpl_to_verts
            from utils.misc import save_trimesh
            from utils.smpl import from_qpos_to_smpl
            inspect_path = f"inspect_out/prox/get_humor_dict_obs_from_sim/{t:03d}/"
            pred_verts, faces = from_qpos_to_smpl(prev_qpos[0], self)
            save_trimesh(pred_verts[0, 0], faces, inspect_path + f"prev_qpos_{t:03d}.ply")
            pred_verts, faces = from_qpos_to_smpl(qpos[0], self)
            save_trimesh(pred_verts[0, 0], faces, inspect_path + f"qpos_{t:03d}.ply")

        # fk_result --> keys: (['qpos', 'wbpos', 'wbquat', 'len'])
        # qpos: body representation (1->3: root position, 3->7: root orientation, 7->end: joint orientation)
        # Rotations are represented in euler angles.
        # wbpos --> world body pose
        # wbquat --> world body quaternion
        # it contains these values for both the previous and current qpos
        fk_result = self.humanoid.qpos_fk(torch.from_numpy(qpos_stack), to_numpy=False, full_return=False)
        trans_batch = torch.from_numpy(trans[None]) # ([1, 2, 3])

        joints = fk_result["wbpos"].reshape(-1, 24, 3)[:, MUJOCO_2_SMPL].reshape(-1, 72)[:, :66]
        pose_aa_mat = aa2mat(torch.from_numpy(pose_aa.reshape(-1, 3))).reshape(1, 2, 24, 4, 4)[..., :3, :3]
        trans_vel, joints_vel, root_orient_vel = estimate_velocities(trans_batch, pose_aa_mat[:, :, 0], joints[None],
                                                                     30, aa_to_mat=False)
        if 0:
            from utils.misc import save_pointcloud
            jts = joints.reshape(-1, 22, 3)
            save_pointcloud(jts[0], inspect_path + f"prev_qpos_jts.ply")
            save_pointcloud(jts[1], inspect_path + f"qpos_jts.ply")

        humor_out = {}
        humor_out['trans_vel'] = trans_vel[:, 0:1, :]
        humor_out['joints_vel'] = joints_vel[:, 0:1, :]
        humor_out['root_orient_vel'] = root_orient_vel[:, 0:1, :]
        humor_out['joints'] = joints[None, 1:2]
        humor_out['pose_body'] = pose_aa_mat[:, 1:2, 1:22]  # contains current qpos
        humor_out['root_orient'] = pose_aa_mat[:, 1:2, 0]
        humor_out['trans'] = trans_batch[:, 1:2]

        ######################## Compute 2D Keypoint projection and 3D keypoint ######################
        grad_frame_num = self.kin_cfg.model_specs.get("grad_frame_num", 1)
        t = self.cur_t + 1
        # selects reference 2D keypoints from off-the-shelf 2D keypoint detector
        joints2d_gt = self.context_dict['joints2d'][t:(t + grad_frame_num)].copy() # (1, 12, 3)

        if joints2d_gt.shape[0] < grad_frame_num:
            joints2d_gt = np.pad(joints2d_gt, ([0, grad_frame_num - joints2d_gt.shape[0]], [0, 0], [0, 0]), mode="edge")

        inliers = joints2d_gt[..., 2] > self.op_thresh # boolean: (1, 12)
        self.hb.update_tgt_joints(joints2d_gt[..., :2], inliers)

        # input_vect contains the SMPL pose corresponding to current qpos only
        input_vec = np.concatenate([humor_out['trans'].numpy(), pose_aa[1:2].reshape(1, -1, 72)], axis=2) # (1, 1, 75)
        
        ######################################## Projection of 3D to 2D keypoints ######################################
        # pred_2d --> (1, 12, 2)
        # cam_pred_3d --> (1, 14, 2)
        # self.hb.proj2d --> projects 3D keypoints to 2D using the camera parameters and SMPL joints format
        data_name = self.kin_cfg.data_name
        pred_2d, cam_pred_3d = self.hb.proj2d(fk_result["wbpos"][1:2].reshape(24, 3).numpy(), return_cam_3d=True,
                                              data_name=data_name)
        ################################################################################################################
        
        humor_out["pred_joints2d"] = torch.from_numpy(pred_2d[None,])
        humor_out["cam_pred_3d"] = torch.from_numpy(cam_pred_3d[None,])

        if self.kin_cfg.model_specs.get("use_tcn", False) and self.kin_cfg.model_specs.get("tcn_3dpos", False):
            cam_pred_tcn_3d = self.pred_tcn['cam_body_pos'][None,]
            humor_out["cam_pred_tcn_3d"] = torch.from_numpy(cam_pred_tcn_3d[None,])

        order = self.kin_cfg.model_specs.get("use_3d_grad_ord", 1)
        normalize = self.kin_cfg.model_specs.get("normalize_grad", False)

        depth = np.mean(cam_pred_3d[..., 2])
        if self.kin_cfg.model_specs.get("use_3d_grad", False):
            num_adpt_grad = 1
            grad_step = self.kin_cfg.model_specs.get("grad_step", 5)
            pose_grad, input_vec_new, curr_loss = self.multi_step_grad(input_vec, order=order, num_adpt_grad=num_adpt_grad, normalize=normalize, step_size=grad_step)
            multi = depth / 10
            pose_grad[:6] *= multi
            humor_out["proj2dgrad"] = pose_grad

        elif self.kin_cfg.model_specs.get("use_3d_grad_line", False):
            proj2dgrad = self.proj_2d_line_loss(input_vec)
            humor_out["proj2dgrad"] = -torch.from_numpy(proj2dgrad)

        elif self.kin_cfg.model_specs.get("use_3d_grad_adpt", False):
            num_adpt_grad = self.kin_cfg.model_specs.get("use_3d_grad_adpt_num", 5)
            grad_step = self.kin_cfg.model_specs.get("grad_step", 5)
            
            ############################################################################################################
            ######################### This is the MPG !!! ##############################################################
            pose_grad, input_vec_new, curr_loss = self.multi_step_grad(input_vec,
                                                                       order=order,
                                                                       num_adpt_grad=num_adpt_grad,
                                                                       normalize=normalize,
                                                                       step_size=grad_step)
            ############################################################################################################

            multi = depth / 10
            pose_grad[:6] *= multi
            humor_out["proj2dgrad"] = pose_grad
            # also defined here:
            # humor_out["pred_joints2d"]
            # humor_out["cam_pred_3d"]
            if 0:
                from utils.misc import plot_joints_cv2
                # black = np.zeros([1080, 1920, 3], dtype=np.uint8)
                black = np.zeros([900, 900, 3], dtype=np.uint8)
                pred_joints2d = humor_out["pred_joints2d"]
                plot_joints_cv2(black, pred_joints2d[0], show=True, with_text=True, sc=3)

        return humor_out

    def geo_trans(self, input_vec_new):
        delta_t = np.zeros(3)
        geo_tran_cap = self.kin_cfg.model_specs.get("geo_trans_cap", 0.1)
        try:
            inliners = self.hb.inliers
            if np.sum(inliners) >= 3:
                wbpos = self.hb.fk_batch_grad(input_vec_new)
                cam2d, cam3d = self.hb.proj2d(wbpos, True)
                cam3d = smpl_op_to_op(cam3d)
                j2ds = self.hb.gt_2d_joints[0].copy()
                K = self.camera_params['K']
                fu, fv, cu, cv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
                A1 = np.tile([fu, 0], 12)
                A2 = np.tile([0, fv], 12)
                A3 = np.tile([cu, cv], 12) - j2ds.flatten()

                b_1 = j2ds[:, 0] * cam3d[0, :, 2] - fu * cam3d[0, :, 0] - cu * cam3d[0, :, 2]
                b_2 = j2ds[:, 1] * cam3d[0, :, 2] - fv * cam3d[0, :, 1] - cv * cam3d[0, :, 2]
                b = np.hstack([b_1[:, None], b_2[:, None]]).flatten()[:, None]
                A = np.hstack([A1[:, None], A2[:, None], A3[:, None]])
                A = A[np.tile(inliners, 2).squeeze()]
                b = b[np.tile(inliners, 2).squeeze()]
                u, sigma, vt = np.linalg.svd(A)

                Sigma_pinv = np.zeros_like(A).T
                Sigma_pinv[:3, :3] = np.diag(1 / sigma)
                delta_t = vt.T.dot(Sigma_pinv).dot(u.T).dot(b)
                delta_t = self.hb.full_R.T @ delta_t[:, 0]

                if np.linalg.norm(delta_t) > geo_tran_cap:
                    delta_t = delta_t / np.linalg.norm(delta_t) * geo_tran_cap
            else:
                delta_t = np.zeros(3)

        except Exception as e:
            print("error in svd and pose", e)
        return delta_t

    # def geo_trans(self, input_vec_new):
    #     best_delta_t = np.zeros((3, 1))
    #     geo_tran_cap = self.kin_cfg.model_specs.get("geo_trans_cap", 0.01)
    #     wbpos = self.hb.fk_batch_grad(input_vec_new)
    #     cam2d, cam3d = self.hb.proj2d(wbpos, True)
    #     cam3d = smpl_op_to_op(cam3d)

    #     # j2ds = self.hb.gt_2d_joints[0].copy()
    #     # inliners = self.hb.inliers

    #     grad_frame_num = self.kin_cfg.model_specs.get("grad_frame_num", 1)
    #     t = self.cur_t + 1
    #     joints2d_gt = self.context_dict['joints2d'][t:(t +
    #                                                    grad_frame_num)].copy()
    #     if joints2d_gt.shape[0] < grad_frame_num:
    #         joints2d_gt = np.pad(
    #             joints2d_gt,
    #             ([0, grad_frame_num - joints2d_gt.shape[0]], [0, 0], [0, 0]),
    #             mode="edge")

    #     j2ds = joints2d_gt[..., :2].copy()[0]
    #     # inliners = joints2d_gt[..., 2].copy() > 0.5

    #     best_err = np.inf
    #     for _ in range(self.n_ransac):
    #         samples = np.arange(j2ds.shape[0])
    #         samples = np.random.choice(samples, 3, replace=False)
    #         inds = np.zeros((1, j2ds.shape[0]), dtype=bool)
    #         inds[:, samples] = 1
    #         K = self.camera_params['K']
    #         fu, fv, cu, cv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    #         A1 = np.tile([fu, 0], 12)
    #         A2 = np.tile([0, fv], 12)
    #         A3 = np.tile([cu, cv], 12) - j2ds.flatten()

    #         b_1 = j2ds[:, 0] * cam3d[0, :, 2] - fu * cam3d[
    #             0, :, 0] - cu * cam3d[0, :, 2]
    #         b_2 = j2ds[:, 1] * cam3d[0, :, 2] - fv * cam3d[
    #             0, :, 1] - cv * cam3d[0, :, 2]
    #         b_orig = np.hstack([b_1[:, None], b_2[:, None]]).flatten()[:, None]
    #         A_orig = np.hstack([A1[:, None], A2[:, None], A3[:, None]])
    #         A = A_orig[np.tile(inds, 2).squeeze()].copy()
    #         b = b_orig[np.tile(inds, 2).squeeze()].copy()
    #         try:
    #             u, sigma, vt = np.linalg.svd(A)
    #             Sigma_pinv = np.zeros_like(A).T
    #             Sigma_pinv[:3, :3] = np.diag(1 / sigma)
    #             delta_t = vt.T.dot(Sigma_pinv).dot(u.T).dot(b)
    #             err = np.linalg.norm(A_orig.dot(delta_t) - b_orig, 2)
    #             if err < best_err:
    #                 best_err = err
    #                 best_delta_t = delta_t
    #         except Exception:
    #             continue

    #     delta_t = self.hb.full_R.T @ best_delta_t[:, 0]

    #     meter_cap = geo_tran_cap
    #     if np.linalg.norm(delta_t) > meter_cap:
    #         delta_t = delta_t / np.linalg.norm(delta_t) * meter_cap
    #     return delta_t

    def multi_step_grad(self, input_vec, num_adpt_grad=5, normalize=False, order=2, step_size=5):
        geo_trans = self.kin_cfg.model_specs.get("geo_trans", False)
        tcn_root_grad = self.kin_cfg.model_specs.get("tcn_root_grad", False)
        input_vec_new = input_vec.copy()
        #########
        # this loss is important
        prev_loss = orig_loss = self.hb.proj_2d_loss(input_vec_new, ord=order, normalize=normalize)
        #############
        if tcn_root_grad:
            world_body_pos, world_trans = self.pred_tcn['world_body_pos'], self.pred_tcn['world_trans']
            pred_root_vec = sRot.from_matrix(op_to_root_orient(world_body_pos[None,])).as_rotvec()  # tcn's root
            input_vec_new[..., 3:6] = pred_root_vec

        if order == 1:
            step_size = 0.00001
            step_size_a = step_size * np.clip(prev_loss, 0, 5)
        else:
            if normalize:
                step_size_a = step_size / 1.02
            else:
                step_size_a = 0.000005
        for iteration in range(num_adpt_grad):
            # it enters the if
            if self.kin_cfg.model_specs.get("use_3d_grad_sept", False):
                proj2dgrad_body = self.proj_2d_body_loss(input_vec_new, ord=order, normalize=normalize)
                proj2dgrad = self.proj_2d_loss(input_vec_new, ord=order, normalize=normalize)
                proj2dgrad[..., 3:] = proj2dgrad_body[..., 3:]
                proj2dgrad = np.nan_to_num(proj2dgrad, posinf=0, neginf=0)  # This is essentail, otherwise nan will get more
            else:
                proj2dgrad = self.proj_2d_loss(input_vec_new, ord=order, normalize=normalize)
                proj2dgrad = np.nan_to_num(proj2dgrad, posinf=0, neginf=0)  # This is essentail, otherwise nan will get more

            # import ipdb
            # ipdb.set_trace()
            # wbpos = self.hb.fk_batch_grad(input_vec_new); pred_joints2d = self.hb.proj2d(wbpos); joblib.dump(pred_joints2d, "a.pkl"); joblib.dump(self.hb.gt_2d_joints, "b.pkl")

            input_vec_new = input_vec_new - proj2dgrad * step_size_a

            if geo_trans:
                delta_t = self.geo_trans(input_vec_new)
                delta_t = np.concatenate([delta_t, np.zeros(72)])
                input_vec_new += delta_t

            curr_loss = self.hb.proj_2d_loss(input_vec_new, ord=order, normalize=normalize)

            if curr_loss > prev_loss:
                step_size_a *= 0.5
            prev_loss = curr_loss

        if self.hb.proj_2d_loss(input_vec_new, ord=order, normalize=normalize) > orig_loss:
            pose_grad = torch.zeros(proj2dgrad.shape)
        else:
            pose_grad = torch.from_numpy(input_vec_new - input_vec)
        return pose_grad, input_vec_new, curr_loss

    def load_camera_params(self):
        if "scene_name" in self.context_dict:
            scene_key = self.context_dict['scene_name']
        else:
            scene_key = self.context_dict['seq_name'][:-9]

        prox_path = self.kin_cfg.data_specs['prox_path']

        with open(f'{prox_path}/calibration/Color.json', 'r') as f:
            cameraInfo = json.load(f)
            K = np.array(cameraInfo['camera_mtx']).astype(np.float32)

        with open(f'{prox_path}/cam2world/{scene_key}.json', 'r') as f:
            camera_pose = np.array(json.load(f)).astype(np.float32)
            R = camera_pose[:3, :3]
            tr = camera_pose[:3, 3]
            R = R.T
            tr = -np.matmul(R, tr)

        with open(f'{prox_path}/alignment/{scene_key}.npz', 'rb') as f:
            aRt = np.load(f)
            aR = aRt['R']
            atr = aRt['t']
            aR = aR.T
            atr = -np.matmul(aR, atr)

        full_R = R.dot(aR)
        full_t = R.dot(atr) + tr

        if self.cc_cfg.ignore_align:
            full_R = R
            full_t = tr

        self.camera_params = {"K": K, "R": R, "tr": tr, "aR": aR, "atr": atr, "full_R": full_R, "full_t": full_t}
        self.camera_params_torch = {k: torch.from_numpy(v).double() for k, v in self.camera_params.items()}

    def step(self, a, kin_override=False):
        # a here is action --> shape(114, )
        fail = False
        cfg = self.kin_cfg
        cc_cfg = self.cc_cfg

        self.prev_qpos = self.get_humanoid_qpos()
        # self.prev_qpos = self.get_expert_qpos() ## No simulate

        self.prev_qvel = self.get_humanoid_qvel()
        self.prev_bquat = self.bquat.copy()
        self.prev_hpos = self.get_head().copy()

        ########################## important function ##########################
        # here the kinematic target pose is updated with the residuals that come
        # from the kinematic policy.
        # this step_ar is related to self.target['wbpos'],
        # here it sets self.target = fk_res
        self.step_ar(a.copy()) # a is action
        ########################################################################

        # if flags.debug:
        # self.target = self.humanoid.qpos_fk(torch.from_numpy(self.context_dict['qpos'][self.cur_t:self.cur_t + 1])) # GT
        # self.target = self.smpl_humanoid.qpos_fk(self.ar_context['ar_qpos'][self.cur_t + 1]) # Debug
        # self.target = self.humanoid.qpos_fk(torch.from_numpy(self.gt_targets['qpos'][self.cur_t:self.cur_t + 1])) # GT
        # self.target = self.humanoid.qpos_fk(torch.from_numpy(self.gt_targets['qpos'][self.cur_t:self.cur_t + 1])) # Use gt

        ############################# UHC step ########################################################
        # gets the obs that DOES NOT contain scene voxels
        cc_obs = self.get_cc_obs() # runs super().get_obs()
        cc_obs = self.cc_running_state(cc_obs, update=False)
        ############ CC step #########################################################################
        # it is the PolicyGaussian, dunno why gaussian. Is it the kin policy? or the UHC policy?
        # NUK: as cc_a is the input to the simulation I assume that this is the UHC policy
        cc_a = self.cc_policy.select_action(torch.from_numpy(cc_obs)[None,], mean_action=True)[0].numpy()
        ##############################################################################################

        ################################ Physical simulation occurs here ##############################################
        if flags.debug:
            self.do_simulation(cc_a, self.frame_skip)
            # self.data.qpos[:self.qpos_lim] = self.gt_targets['qpos'][self.cur_t + 1]  # debug
            # self.sim.forward()  # debug
        else:
            ###### normal operation
            if kin_override:
                self.data.qpos[:self.qpos_lim] = self.gt_targets['qpos'][self.cur_t + 1]  # debug
                self.sim.forward()  # debug
            else:
                # it comes here by default
                if self.simulate:
                    try:
                        self.do_simulation(cc_a, self.frame_skip)
                    except Exception as e:
                        print("Exception in do_simulation", e, self.cur_t)
                        fail = True
                else:
                    self.data.qpos[:self.qpos_lim] = self.get_expert_qpos()  # debug
                    self.sim.forward()  # debug


        if 0:
            from utils.smpl import smpl_to_verts
            from utils.smpl import from_qpos_to_smpl
            from utils.misc import save_trimesh
            from utils.misc import save_pointcloud
            t = self.cur_t
            target = self.target
            qpos = target['qpos']
            trans = target["trans"]
            inspect_path = f"inspect_out/single_agent/step/chi3d/"
            # qpos --> converted to smpl is correct
            qpos_v, faces = from_qpos_to_smpl(qpos[0], self) # input has to be (76,)
            save_trimesh(qpos_v[0, 0], faces, inspect_path + f"{t:02d}/qpos.ply")
            # wbpos --> it is correct, these are joints, not smpl params!
            simu_qpos = self.data.qpos
            qpos_v, faces = from_qpos_to_smpl(simu_qpos, self) # input has to be (76,)
            save_trimesh(qpos_v[0, 0], faces, inspect_path + f"{t:02d}/simu_qpos.ply")
            #
        # if self.cur_t == 0 and self.agent.global_start_fr == 0:
        #     # ZL: Stablizing the first frame jump
        #     self.data.qpos[:self.qpos_lim] = self.get_expert_qpos()  # debug
        #     self.data.qvel[:] = 0
        #     self.sim.forward()  # debug
        ##############################################################################################

        self.cur_t += 1
        self.bquat = self.get_body_quat()
        # get obs
        reward = 1.0
        if cfg.env_term_body == 'body':
            body_diff = self.calc_body_diff()
            if self.mode == "train":
                body_gt_diff = self.calc_body_gt_diff()
                fail = fail or (body_diff > 2.5 or body_gt_diff > 3.5)
            else:
                if cfg.ignore_fail:
                    fail = False
                else:
                    fail = fail or body_diff > 7


                # if fail:
                #     from pathlib import Path
                #     from utils.misc import save_pointcloud
                #     ## This is self.data.body_xpos[1:self.body_lim]
                #     cur_wbpos = self.get_wbody_pos().reshape(-1, 3)
                #     ### THIS is self.target['wbpos'], it is somehow related to self.pred_tcn
                #     # it is also related to next_global_out --> net that uses motion prior (UHMModel)
                #     # so this is realted to the funcs: step_ar() and load_context()
                #     e_wbpos = self.get_expert_joint_pos().reshape(-1, 3)
                #     ####
                #     save_pointcloud(cur_wbpos, f"inspect_out/sim_render/chi3d/humanoid_kin_res/cur_wbpos.ply")
                #     save_pointcloud(e_wbpos, f"inspect_out/sim_render/chi3d/humanoid_kin_res/e_wbpos.ply")
                #     for cam_i in range(1, 5):
                #         img = self.sim.render(width=400, height=400, camera_name=f"camera{cam_i}")
                #         # plot(img)
                #         # img = rot_img(img, show=False)
                #         path = Path(f"inspect_out/sim_render/chi3d/humanoid_kin_res/sim_cam{cam_i}/%07d.png" % self.cur_t)
                #         path.parent.mkdir(parents=True, exist_ok=True)
                #         save_img(path, img)

            # fail = body_diff > 10
            # print(fail, self.cur_t)
            # fail = False
        else:
            raise NotImplemented()
        # if flags.debug:
        #     fail = False
        end = (self.cur_t >= cc_cfg.env_episode_len) or (self.cur_t + self.start_ind >= self.context_dict['len'])
        done = fail or end
        # if done:
        # print(f"Fail: {fail} | End: {end}", self.cur_t, body_diff)
        percent = self.cur_t / self.context_dict['len']

        ############################## The new observation is computed here #########################################
        if not done:
            # NUK: this one calls self.get_ar_obs_v1() and it is different from self.get_cc_obs() which calls
            # self.get_full_obs_v2()()
            # NOTE: MPG is called inside this get_obs() function
            obs = self.get_obs()  # can no longer compute obs when done....
        else:
            obs = np.zeros(self.obs_dim)
        #############################################################################################################

        return obs, reward, done, {'fail': fail, 'end': end, "percent": percent}

    def set_mode(self, mode):
        self.mode = mode

    def ar_fail_safe(self):
        self.data.qpos[:self.qpos_lim] = self.context_dict['ar_qpos'][self.cur_t + 1]
        # self.data.qpos[:self.qpos_lim] = self.get_target_qpos()
        self.data.qvel[:self.qvel_lim] = self.context_dict['ar_qvel'][self.cur_t + 1]
        self.sim.forward()

    def reset_model(self, qpos=None, qvel=None):
        cfg = self.kin_cfg
        ind = 0
        self.start_ind = 0
        if qpos is None:
            init_pose_aa = self.context_dict['init_pose_aa']
            init_trans = self.context_dict['init_trans']
            init_qpos = smpl_to_qpose(torch.from_numpy(init_pose_aa[None,]), self.model, torch.from_numpy(init_trans[None,]), count_offset=True).squeeze()
            init_vel = np.zeros(self.qvel_lim)
        else:
            init_qpos = qpos
            init_vel = qvel
        #######################
        # I think that here the pose is updated to the init pose
        if 0:
            sim = mujoco_py.MjSim(self.model)
            from mujoco_py import MjViewer
            viewer = MjViewer(sim)
            viewer.render()

        self.set_state(init_qpos, init_vel)

        if 0:
            sim.step()
            viewer.render()

        #######################
        self.prev_qpos = self.get_humanoid_qpos()

        ################################### GET OBS #################################
        obs = self.get_obs()
        #############################################################################

        if 0:
            sim.step()
            viewer.render()

        return obs

    def viewer_setup(self, mode):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.lookat[:2] = self.get_humanoid_qpos()[:2]
        # if mode not in self.set_cam_first:
        #     self.viewer.video_fps = 33
        #     self.viewer.frame_skip = self.frame_skip
        #     self.viewer.cam.distance = self.model.stat.extent * 1.2
        #     self.viewer.cam.elevation = -20
        #     self.viewer.cam.azimuth = 45
        #     self.set_cam_first.add(mode)

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == "human":
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == "rgb_array" or mode == "depth_array":
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)

            self._viewers[mode] = self.viewer
        self.viewer_setup("rgb")

        full_R, full_t = self.camera_params['full_R'], self.camera_params['full_t']
        distance = np.linalg.norm(full_t)
        x_axis = full_R.T[:, 0]
        pos_3d = -full_R.T.dot(full_t)
        rotation = sRot.from_matrix(full_R).as_euler("XYZ", degrees=True)
        self.viewer.cam.distance = 2  # + 3 to have better viewing
        self.viewer.cam.lookat[:] = pos_3d + x_axis
        self.viewer.cam.azimuth = 90 - rotation[2]
        self.viewer.cam.elevation = -8

        return self.viewer

    def match_heading_and_pos(self, qpos_1, qpos_2):
        posxy_1 = qpos_1[:2]
        qpos_1_quat = self.remove_base_rot(qpos_1[3:7])
        qpos_2_quat = self.remove_base_rot(qpos_2[3:7])
        heading_1 = get_heading_q(qpos_1_quat)
        qpos_2[3:7] = de_heading(qpos_2[3:7])
        qpos_2[3:7] = quaternion_multiply(heading_1, qpos_2[3:7])
        qpos_2[:2] = posxy_1
        return qpos_2

    def get_expert_qpos(self, delta_t=0):
        expert_qpos = self.target['qpos'].copy().squeeze()
        return expert_qpos

    def get_target_kin_pose(self, delta_t=0):
        return self.get_expert_qpos()[7:]

    def get_expert_joint_pos(self, delta_t=0):
        # world joint position
        wbpos = self.target['wbpos'].squeeze()
        return wbpos

    def get_expert_com_pos(self, delta_t=0):
        # body joint position
        body_com = self.target['body_com'].squeeze()
        return body_com

    def get_expert_bquat(self, delta_t=0):
        bquat = self.target['bquat'].squeeze()
        return bquat

    def get_expert_wbquat(self, delta_t=0):
        wbquat = self.target['wbquat'].squeeze()
        return wbquat

    def get_expert_shape_and_gender(self):
        cfg = self.cc_cfg

        shape = self.context_dict['beta'][0].squeeze()
        if shape.shape[0] == 10:
            shape = np.concatenate([shape, np.zeros(6)])

        gender = self.context_dict['gender'][0].squeeze()
        obs = []
        if cfg.get("has_pca", True):
            obs.append(shape)

        obs.append([gender])

        if cfg.get("has_weight", False):
            obs.append([self.weight])

        if cfg.get("has_bone_length", False):
            obs.append(self.smpl_robot.bone_length)

        return np.concatenate(obs)

    def calc_body_diff(self):
        cur_wbpos = self.get_wbody_pos().reshape(-1, 3)
        e_wbpos = self.get_expert_joint_pos().reshape(-1, 3)
        diff = cur_wbpos - e_wbpos
        diff *= self.jpos_diffw
        jpos_dist = np.linalg.norm(diff, axis=1).sum()
        return jpos_dist

    def calc_body_ar_diff(self):
        cur_wbpos = self.get_wbody_pos().reshape(-1, 3)
        # e_wbpos = self.get_target_joint_pos().reshape(-1, 3)
        e_wbpos = self.context_dict['ar_wbpos'][self.cur_t + 1].reshape(-1, 3)
        diff = cur_wbpos - e_wbpos
        diff *= self.jpos_diffw
        jpos_dist = np.linalg.norm(diff, axis=1).sum()
        return jpos_dist

    def calc_body_gt_diff(self):
        cur_wbpos = self.get_wbody_pos().reshape(-1, 3)
        e_wbpos = self.gt_targets['wbpos'][self.cur_t].reshape(-1, 3)
        diff = cur_wbpos - e_wbpos
        diff *= self.jpos_diffw
        jpos_dist = np.linalg.norm(diff, axis=1).sum()
        return jpos_dist

    def get_expert_attr(self, attr, ind):
        return self.context_dict[attr][ind].copy()


if __name__ == "__main__":
    pass
