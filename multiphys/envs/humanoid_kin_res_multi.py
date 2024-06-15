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

import os
import os.path as osp
import sys
import json
import copy
import mujoco_py
sys.path.append(os.getcwd())

from uhc.khrylib.rl.envs.common import mujoco_env
from uhc.khrylib.utils import *
from uhc.khrylib.rl.core.policy_gaussian import PolicyGaussian
from uhc.khrylib.rl.core.critic import Value
from uhc.khrylib.models.mlp import MLP
from uhc.models.policy_mcp import PolicyMCP
from uhc.utils.flags import flags
from uhc.envs.humanoid_im import HumanoidEnv

from mujoco_py import functions as mjf
import time
from scipy.linalg import cho_solve, cho_factor
import joblib
import numpy as np
import matplotlib.pyplot as plt
from uhc.smpllib.torch_smpl_humanoid import Humanoid
from uhc.smpllib.smpl_mujoco import smpl_to_qpose_multi
from uhc.smpllib.smpl_mujoco import smpl_to_qpose_torch_multi
from uhc.smpllib.smpl_mujoco import qpos_to_smpl_multi
from uhc.utils.torch_geometry_transforms import (angle_axis_to_rotation_matrix as aa2mat, rotation_matrix_to_angle_axis as mat2aa)

from embodiedpose.models.humor.utils.humor_mujoco import MUJOCO_2_SMPL
from embodiedpose.models.humor.body_model.utils import smpl_to_openpose
from multiphys.smpllib.scene_robot_multi import SceneRobotMulti
from embodiedpose.models.humor.utils.velocities import estimate_velocities
from embodiedpose.models.uhm_model import UHMModel
from scipy.spatial.transform import Rotation as sRot
from uhc.utils.tools import CustomUnpickler
from autograd import elementwise_grad as egrad
from uhc.smpllib.np_smpl_humanoid_batch import Humanoid_Batch
import collections
from uhc.utils.math_utils import normalize_screen_coordinates, op_to_root_orient, smpl_op_to_op
from uhc.utils.torch_ext import isNpArray
from uhc.smpllib.smpl_parser import (
    SMPL_BONE_ORDER_NAMES,
)
from uhc.utils.transformation import (
    quaternion_multiply_batch,
    quaternion_inverse_batch,
)
from uhc.smpllib.smpl_mujoco import SMPLConverter
from utils.misc import save_img
import datetime
from mujoco_py import load_model_from_xml
from lxml.etree import SubElement


np.set_printoptions(precision=3, suppress=True, linewidth=100)
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


def get_xpos(n, body_names, data):
    vf_bodies = body_names[1:-24] if n == 0 else body_names[-24:]
    all_body_xpos = []
    for body in vf_bodies:
        body_xpos = data.get_body_xpos(body)
        all_body_xpos.append(body_xpos)
    return np.concatenate(all_body_xpos)


def plot_voxel(voxelarray):
    import matplotlib.pyplot as plt
    import numpy as np

    # prepare some coordinates
    # x, y, z = np.indices((8, 8, 8))
    # draw cuboids in the top left and bottom right corners, and a link between
    # bolean array of shape (8, 8, 8)
    # voxelarray = abs(x - y) + abs(y - z) + abs(z - x) <= 2

    # set the colors of each object
    colors = np.empty(voxelarray.shape, dtype=object)
    colors[voxelarray] = 'red'
    # and plot everything
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(voxelarray, facecolors=colors, edgecolor='k')
    plt.show()

class MultiHumanoidKinEnvRes(HumanoidEnv):
    """THIS WILL HANDLE MULTIPLE ROBOTS AND ACTUALLY INHERIT FROM HumanoidEnv BUT WILL HAVE
     MODIFIED METHODS
     This class is responsible for MPG, so it uses the 3D projected kpts and compares w/ the
     2D inputs
     """
    # Wrapper class that wraps around Copycat agent from UHC

    def __init__(self, kin_cfg, init_context, cc_iter=-1, mode="train", agent=None):

        # self.num_agents = 2
        self.num_agents = kin_cfg.num_agents
        self.reser_robot_pass = 0
        self.one_agent_dim = 69
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
        # frame_skip = kin_cfg.frame_skip # do not use this, the simu fails if frame_skip>15
        # print(f"*** Using skip frame of: {frame_skip} ***")

        self.load_context_pass = 0
        self.pred_joints2d = [[], []]
        # have al the robots here, so that this can also operate with only one robot
        self.smpl_robot = []

        # env specific
        self.use_quat = cc_cfg.robot_cfg.get("ball", False)
        cc_cfg.robot_cfg['span'] = kin_cfg.model_specs.get("voxel_span", 1.8)

        #### inside this object it this gets the scene! not at init though
        # masterfoot default is False, what does it do?
        # what is the difference btw smpl_robot and smpl_robot_orig?
        # this is used in HumanoidEnv(mujoco_env.MujocoEnv), located at uhc/envs/humanoid_im.py
        self.smpl_robot_orig = SceneRobotMulti(cc_cfg.robot_cfg, data_dir=osp.join(cc_cfg.base_dir, "data/smpl"))
        # here Humanoid batch is also use, but what is it? not that only one agent is used here to set the offsets!
        self.hb = [Humanoid_Batch(data_dir=osp.join(cc_cfg.base_dir, "data/smpl")) for _ in range(self.num_agents)]

        ############################# Agent loaded here #########################################
        # the important function is located in Robot class --> self.load_from_skeleton() where
        # SkeletonMesh() is instantiated
        # what is masterfoot?
        # self.smpl_robot = SceneRobotMulti(
        #     cc_cfg.robot_cfg,
        #     data_dir=osp.join(cc_cfg.base_dir, "data/smpl"),
        #     masterfoot=cc_cfg.masterfoot,
        #     num_agent=1
        # )
        # todo, dont add the second agent here, do it inside smpl_robot, just like add_simple_scene
        # but NOTE: the second robot has to be added before the simu is started, right?
        # maybe not necessarily becuase the scene can also be added afterwards
        #
        # self.smpl_robot2 = SceneRobotMulti(
        #     cc_cfg.robot_cfg,
        #     data_dir=osp.join(cc_cfg.base_dir, "data/smpl"),
        #     masterfoot=cc_cfg.masterfoot,
        #     num_agent=1
        # )

        self.smpl_robot = [SceneRobotMulti(
            cc_cfg.robot_cfg,
            data_dir=osp.join(cc_cfg.base_dir, "data/smpl"),
            masterfoot=cc_cfg.masterfoot,
            num_agent=1
        ) for _ in range(self.num_agents)]
        ##########################################################################################
        if self.num_agents > 1:
            # self.merge_agents_xml(self.smpl_robot[0], self.smpl_robot[1])
            # better pass the list of robots
            self.merge_agents_xml(self.smpl_robot)

        # this xml_str specifies only the human robot, lightning and cameras, and maybe the floor also
        # here each part of the body mesh for the simu is defined following the kinematic tree
        # also actuators are assigned to each joint.
        # probably I have to append the additional agent/robot in this xml_str
        ## This XML already contains the 2 agents ##
        self.xml_str = self.smpl_robot[0].export_xml_string().decode("utf-8") # here xml create w/out betas info
        # self.xml_str2 = self.smpl_robot2.export_xml_string().decode("utf-8")
        if 0:
            from utils.misc import xml_str_to_file
            xml_str_to_file(self.xml_str, "inspect_out/xml_s/meged_xml.xml")
            xml_str_to_file(self.xml_str, "inspect_out/xml_s/meged_xml_swap.xml")
            fname = f"inspect_out/xml_s/robot1.xml"
            self.smpl_robot.write_xml(fname)
            fname = "inspect_out/xml_s/robot2.xml"
            self.smpl_robot2.write_xml(fname)
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
            # NUK: i made this change
            self.j2d_seq_feat = [collections.deque([0] * self.num_context, self.num_context) for i in range(2)]

        self.body_grad = np.zeros(63)
        self.bm = bm = self.motion_prior.bm_dict['neutral']
        self.smpl2op_map = smpl_to_openpose(bm.model_type, use_hands=False, use_face=False, use_face_contour=False,
                                            openpose_format='coco25') # this is used by the Humanoid_Batch class I think
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

        mujoco_env.MujocoEnv.__init__(self, self.xml_str, frame_skip=15)

        self.prev_qpos = self.data.qpos.copy() # (76, )--> this is now (152,)
        # qpos is one per body in the xml, but
        # self.data.ctrl is (69,) and it is one per actuator
        # self.model.actuator_names
        # self.model.body_names # is tuple of (25)
        # self.model.joint_names # is tuple of (70)
        # self.model.nq # is 76
        # self.model._body_name2id # is 25, maybe it is 25*3=75 because it is 3 per body, but then why is nq=76? and not 75?
        # self.model.dof_bodyid # is 150 for two agents, 75 for one agent

        # todo ojo con esta porque contiene smpl_robot_orig y hay que hacer algo quiza
        self.setup_constants(cc_cfg, cc_cfg.data_specs, mode=mode, no_root=False)
        self.neutral_path = self.kin_cfg.data_specs['neutral_path']
        self.neutral_data = joblib.load(self.neutral_path)
        ###############################
        # this calls self.reset_robot() and SceneRobot.load_from_skeleton() is called from there
        if 0:
            fname = "inspect_out/xml_s/robot2_HumanoidKinEnvRes_init.xml"
            self.smpl_robot2.write_xml(fname)

        # this contains reset_robot(), and in turn it contains load_from_skeleton() and now self.smpl_robot.add_agent()
        # calling this will add the second agent, but is it this the best way to do it?
        self.load_context(init_context) # here betas info should be added to the xml file
        ###############################
        # seems important: defines dimensions of the action space and thus of the policy net
        # I think this should be done per agent
        self.set_action_spaces()
        #  MPG is called inside this function by calling get_humor_dict_obs_from_sim
        self.set_obs_spaces()
        # it seems that this weight is never used, in thatcase this should be multi
        self.weight = mujoco_py.functions.mj_getTotalmass(self.model)
        # all_funcs = mujoco_py.functions.__dict__


        ''' Load CC Controller '''
        # self.get_cc_obs() now contains a list for all agents
        cc_obs = self.get_cc_obs()[0]
        self.state_dim = state_dim = cc_obs.shape[0]
        cc_action_dim = self.action_dim # originally 315, should be 315*2=630? -->maybe not as this sets
        # the action dim for policy net, we want to pass the policy twice, one for each agent
        # selects cc_policy
        if cc_cfg.actor_type == "gauss":
            self.cc_policy = PolicyGaussian(cc_cfg, action_dim=cc_action_dim, state_dim=state_dim)
        elif cc_cfg.actor_type == "mcp":
            self.cc_policy = PolicyMCP(cc_cfg, action_dim=cc_action_dim, state_dim=state_dim)

        self.cc_value_net = Value(MLP(state_dim, cc_cfg.value_hsize, cc_cfg.value_htype))
        if cc_iter != -1:
            cp_path = '%s/iter_%04d.p' % (cc_cfg.model_dir, cc_iter)
        else:
            try:
                cc_iter = np.max([int(i.split("_")[-1].split(".")[0]) for i in os.listdir(cc_cfg.model_dir)])
            except:
                assert len(os.listdir(cc_cfg.model_dir))>0, "ERROR: No checkpoint found. Probably need to add UHC checkpoint 'iter_5700.p' to: './results/motion_im/uhc_explicit/models' "
            cp_path = '%s/iter_%04d.p' % (cc_cfg.model_dir, cc_iter)

        print((f'loading model from checkpoint in {__name__}: %s' % cp_path))
        model_cp = CustomUnpickler(open(cp_path, "rb")).load()
        self.cc_running_state = model_cp['running_state']
        # loads checkpoint
        self.cc_policy.load_state_dict(model_cp['policy_dict'])
        self.cc_value_net.load_state_dict(model_cp['value_dict'])

        # Contact modelling
        # todo multi: change this to handle multiple agents
        # the place were this is used, it done on a per agent basis, so prob not necessary to make multi
        body_id_list = self.model.geom_bodyid.tolist()
        self.contact_geoms = [body_id_list.index(self.model._body_name2id[body]) for body in SMPL_BONE_ORDER_NAMES]
        # would this be useful?
        # SMPL_BONE_ORDER_NAMES2 = [name + "_2" for name in SMPL_BONE_ORDER_NAMES]
        # self.contact_geoms2 = [body_id_list.index(self.model._body_name2id[body]) for body in SMPL_BONE_ORDER_NAMES2]

    # def set_smpl_robot2(self, smpl_robot):
    #     self.smpl_robot2 = smpl_robot

    def setup_constants(self, cfg, data_specs, mode, no_root):
        """ Overrides function from HumanoidEnv"""
        # NUK: this function was added here by me, it overrides method from HumanoidEnv
        self.cc_cfg = cfg
        self.set_cam_first = set()
        # todo. maybe i'll have to modify smpl_robot_orig to contain 2 agents
        self.smpl_model = load_model_from_xml(self.smpl_robot_orig.export_xml_string().decode("utf-8"))

        # if self.cc_cfg.masterfoot:
        #     self.sim_model = load_model_from_path(cfg.mujoco_model_file)
        # else:
        #     self.sim_model = load_model_from_xml(
        #         self.smpl_robot.export_xml_string().decode("utf-8")
        #     )
        # this is already for 2 agents, so sim_model contains both
        self.sim_model = load_model_from_xml(self.smpl_robot[0].export_xml_string().decode("utf-8"))
        self.expert = None
        self.base_rot = data_specs.get("base_rot", [0.7071, 0.7071, 0.0, 0.0])
        self.netural_path = data_specs.get("neutral_path", "sample_data/standing_neutral.pkl")
        self.no_root = no_root
        self.body_diff_thresh = cfg.get("body_diff_thresh", 0.5)
        self.body_diff_thresh_test = cfg.get("body_diff_thresh_test", 0.5)
        # self.body_diff_thresh_test = cfg.get("body_diff_thresh_test", 0.5)
        self.mode = mode
        self.end_reward = 0.0
        self.start_ind = 0
        self.rfc_rate = 1 if not cfg.rfc_decay else 0
        self.prev_bquat = None
        self.load_models()
        self.set_model_base_params()
        self.bquat = self.get_body_quat()
        self.humanoid = [Humanoid(model=self.model), Humanoid(model=self.model)]
        self.curr_vf = None  # storing current vf
        self.curr_torque = None  # Strong current applied torque at each joint
        
    def merge_agents_xml(self, smpl_robot_list):
        """ add a second agent to the xml tree
        here tree2 is from agent2 and self.tree should be from agent1
        """
        # check the xml file
        if 0:
            xml_str = etree.tostring(tree, pretty_print=True).decode("utf-8")
            print(xml_str)

        smpl_robot, smpl_robot2 = smpl_robot_list
        tree = smpl_robot.tree
        tree_agent1_copy = copy.deepcopy(smpl_robot.tree)
        # generate copy of the tree but with modified names according to agent id
        remove_elements = ["equality"]
        tree2, xml_str = smpl_robot2.export_robot_str(remove_elements=remove_elements, agent_id=2)
        tree_agent2 = copy.deepcopy(tree2)
        tree_agent2_copy = copy.deepcopy(tree2)

        if 0:
            fname = "inspect_out/xml_s/add_agent_v2.xml"
            tree2.write(fname, pretty_print=True)
            fname = "inspect_out/xml_s/add_agent_v2_robot1.xml"
            tree.write(fname, pretty_print=True)

        # the agent specification will be added to worldbody node
        worldbody = tree.getroot().find("worldbody")

        agent2_asset = tree_agent2.getroot().find("asset")
        tree1_asset = tree.getroot().find("asset")
        for asset in agent2_asset:
            tree1_asset.append(asset)
            # print(asset.tag)

        if 0:
            from pathlib import Path
            fname = "inspect_out/xml_s/merge_files/asset.xml"
            Path(fname).parent.mkdir(parents=True, exist_ok=True)
            tree.write(fname, pretty_print=True)

        # mujoco_node.append(agent2_asset)

        agent2_body_node = tree_agent2.getroot().find("worldbody").find("body")
        # append body node
        worldbody.append(agent2_body_node)
        # append contact node
        agent2_contacts = tree_agent2.getroot().find("contact")
        tree1_contacts = tree.getroot().find("contact")
        for contact in agent2_contacts:
            tree1_contacts.append(contact)
            # print(asset.tag)

        # exclude pelvis between agents
        # excluding contacts: https://github.com/deepmind/mujoco/issues/104
        # NOTE: it seems insuficient to only exclude contacts between pelvises, maybe need to exclude
        # all pairs of contacts
        if self.kin_cfg.exclude_contacts:
            contact_node = tree1_contacts
            agent1_body_nodes_list = tree_agent1_copy.getroot().find("worldbody").findall(".//body")
            agent2_body_nodes_list = tree_agent2_copy.getroot().find("worldbody").findall(".//body")
            assert len(agent1_body_nodes_list) == len(agent2_body_nodes_list) == 24, "Expecting 24 body nodes when excluding contacts"
            for node1 in agent1_body_nodes_list[:24]:
                for node2 in agent2_body_nodes_list:
                    SubElement(
                        contact_node,
                        "exclude",
                        {
                            # "name": "roots",
                            "body1": f"{node1.attrib['name']}",
                            "body2": f"{node2.attrib['name']}",
                        },
                    )
            print("** Excluding pelvis between agents")

        # mujoco_node.append(agent2_contacts)
        # append actuator node
        agent2_actuator = tree_agent2.getroot().find("actuator")
        tree1_actuators = tree.getroot().find("actuator")
        for actuator in agent2_actuator:
            tree1_actuators.append(actuator)
        # mujoco_node.append(agent2_actuator)
        print('* Merged xml files!')
        if 0:
            fname = "inspect_out/xml_s/merge_files/modif_robot.xml"
            tree.write(fname, pretty_print=True)

    def reset_robot(self):
        self.reser_robot_pass += 1

        for n, context in enumerate(self.context_dict):
            beta = context["beta"].copy()
            gender = context["gender"].copy()
            # this seems important for loading the scene, for chi3d this is 's'-->why?
            if isinstance(context['cam'], list):
                scene_name = context['cam'][0]['scene_name']
            else:
                scene_name = context['cam']['scene_name']

            if 0:
                beta[0] == beta[1]
                from utils.misc import plot_joints_cv2
                joints2d = context["joints2d"]
                joints2d_np = joints2d[0]
                black = np.zeros((900, 900, 3), dtype=np.uint8)
                plot_joints_cv2(black, joints2d_np[None], with_text=True, show=True, sc=2)


                from utils.body_model import pose_to_vertices as pose_to_vertices_
                import smplx
                from functools import partial
                from utils.misc import save_trimesh
                local_bm = smplx.create("data", 'smpl', use_pca=False, batch_size=1)
                pose_to_vertices = partial(pose_to_vertices_, pose_type="smpl", alpha=1, bm=local_bm)
                pose = context['init_pose_aa']
                transl = context['init_trans']
                pose = np.concatenate([pose, transl])
                pose = torch.from_numpy(pose[None, None]).float()
                betas = torch.from_numpy(beta[None, 0, :10]).float()
                verts = pose_to_vertices(pose, betas=betas)
                inspect_path = f"inspect_out/betas/chi3d/reset_robot/{self.reser_robot_pass}/"
                save_trimesh(verts[0,0], local_bm.faces, inspect_path+f"pose_{n}.ply")



            if "obj_info" in context:
                obj_info = context['obj_info']
                self.smpl_robot[0].load_from_skeleton(torch.from_numpy(beta[0:1, :]).float(), gender=gender, obj_info=obj_info)
            else:
                # this seems important for loading the scene
                if not context.get("load_scene", True):
                    scene_name = None
                ####################################################################################################
                # Scene loading with the function add_simple_scene is done here
                # loads humanoid and simulation environment from template file and modifies it according to beta and gender
                if n==0:
                    self.smpl_robot[0].load_from_skeleton(torch.from_numpy(beta[0:1, :]).float(), gender=gender,
                                                       scene_and_key=scene_name, num_agent=1)

                elif n==1:
                    # add second agent from to self.smpl_robot2. here
                    self.smpl_robot[1].load_from_skeleton(torch.from_numpy(beta[0:1, :]).float(), gender=gender,
                                                       scene_and_key=scene_name, num_agent=1)
                else:
                    raise NotImplementedError
                ####################################################################################################
        # NUK: at this points the robots in the xml file have beta info
        # self.merge_agents_xml(self.smpl_robot[0], self.smpl_robot[1])
        self.merge_agents_xml(self.smpl_robot)
        xml_str = self.smpl_robot[0].export_xml_string().decode("utf-8")

        if 0:
            # from utils.misc import write_txt
            fname = "inspect_out/xml_s/humanoidKinEnvRes_reset_robot_1.xml"
            self.smpl_robot.tree.write(fname, pretty_print=True)

            from utils.misc import xml_str_to_file
            xml_str_to_file(self.xml_str, "inspect_out/xml_s/reset_robot/meged_xml.xml")
            xml_str_to_file(self.xml_str, "inspect_out/xml_s/reset_robot/meged_xml_swap.xml")
            # write_txt(fname, xml_str)
        ######################################
        # reloads the simulation using mujoco_py.load_model_from_xml(xml_str) and deletes de old viewers,
        # gets init_qpos and init_qvel
        self.reload_sim_model(xml_str)
        ######################################
        self.weight = self.smpl_robot[0].weight
        # hb is a Humanoid_Batch instance, it contains proj_2d_loss,
        # the np version not the torch one found in the same script
        # BETA here affects the projection later on
        # the beta used here is from the last one set in the for loop!

        self.proj_2d_loss = []
        self.proj_2d_body_loss = []
        self.proj_2d_root_loss = []
        self.proj_2d_line_loss = []

        for n in range(self.num_agents):
            beta = self.context_dict[n]["beta"].copy()
            self.hb[n].update_model(torch.from_numpy(beta[0:1, :16]), torch.tensor(gender[0:1]))
            # todo multi: is smpl2op_map the same for all agents? or needs to be updated for each agent?
            self.hb[n].update_projection(self.camera_params, self.smpl2op_map, MUJOCO_2_SMPL)
            # Losses for the MGP (multi_step_grad) are initialized here!
            # egrad is elementwise_grad from autograd
            self.proj_2d_loss.append(egrad(self.hb[n].proj_2d_loss))
            self.proj_2d_body_loss.append(egrad(self.hb[n].proj_2d_body_loss))
            self.proj_2d_root_loss.append(egrad(self.hb[n].proj_2d_root_loss))
            self.proj_2d_line_loss.append(egrad(self.hb[n].proj_2d_line_loss))

        return xml_str # this return is ignored

    def load_context(self, data_dict):
        # the simulation FOR LOOP starts when load_context_pass=2
        self.load_context_pass += 1
        # todo. I think i should load one context_dict for each agent
        self.context_dict = []
        for data_d in data_dict:
            context_d = {k: v.squeeze().cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in data_d.items()}
            self.context_dict.append(context_d)

        if isinstance(data_dict[0]['cam'], list):
            self.camera_params = data_dict[0]['cam'][0]
        else:
            self.camera_params = data_dict[0]['cam']
        self.camera_params_torch = {k: torch.from_numpy(v).double() if isNpArray(v) else v for k, v in self.camera_params.items()}

        ######################################
        # this does a lot!
        # it reloads the simulation using mujoco_py.load_model_from_xml(xml_str) based on body betas
        # defines the reprojection losses self.proj_2d_loss and adds gradient computations to them
        # this calls self.smpl_robot.load_from_skeleton
        # Here, the BETAS are loaded
        self.reset_robot()

        self.target = []
        self.gt_targets  = []
        self.prev_humor_state  = []
        self.cur_humor_state  = []
        self.pred_tcn  = []

        for n, context in enumerate(self.context_dict):
            
            ######################################
            # this next function update_model() handles var self.body_name
            # todo multi: make multi tambien ojo con esta porque usa model de mujoco
            self.humanoid[n].update_model(self.model, n_agent=n)

            # self.humanoid.update_model(self.model, n_agent=n) # no hacer asi! no sirve
            ######################################
            
            # no need to later update self.context_dict, by doing this we are updating it
            context['len'] = context['pose_aa'].shape[0] - 1
            # todo multi: use smpl_to_qpose_multi!
            gt_qpos = smpl_to_qpose_multi(context['pose_aa'], self.model, trans=context['trans'],
                                          count_offset=True, agent_id=n)

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
                humor_first_pose = context['init_pose_aa'][None,]
                init_trans = context['init_trans'][None,]
                pose = np.concatenate([humor_first_pose, init_trans], axis=1)
                pose = torch.from_numpy(pose).float().cuda()
                verts = pose_to_vertices(pose[None])
                save_trimesh(verts[0,0], local_bm.faces, inspect_path+"humor_first_pose.ply")

                # no trans
                no_trans = np.zeros_like(context['init_trans'][None,])
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
                no_trans = np.zeros_like(context['init_trans'][None,])
                to_floor_no_trans = no_trans - np.array([[0., 0., min_xyz[2]]])
                pose = np.concatenate([humor_first_pose, to_floor_no_trans], axis=1)
                pose = torch.from_numpy(pose).float().cuda()
                verts = pose_to_vertices(pose[None])
                save_trimesh(verts[0,0], local_bm.faces, inspect_path+"humor_first_pose_floor_no_trans.ply")

            # something very hacky to correct the initial translation
            if 0:
                # to_floor_trans = np.array([[-0.581167008413573, -0.6478122600077725, 0.8168622255325317]])
                # context['init_trans'] = to_floor_trans[0]
                context['init_trans'] = to_floor_no_trans[0]

            # this is the HUMOR estimated pose
            # todo modify for 2 people
            init_qpos = smpl_to_qpose_multi(context['init_pose_aa'][None,], self.model,
                                      trans=context['init_trans'][None,],
                                      count_offset=True, agent_id=n)
            context["qpos"] = gt_qpos

            # uses as input  the initial HUMOR estimate, first pose, it serves as first target pose
            # multi: this target should have 2 people
            target = self.humanoid[n].qpos_fk(torch.from_numpy(init_qpos))

            # contains the keys (['trans', 'root_orient', 'pose_body', 'joints', 'root_orient_vel', 'joints_vel'])
            prev_humor_state = {k: data_dict[n][k][:, 0:1, :].clone() for k in self.motion_prior.data_names}
            cur_humor_state = prev_humor_state
            #####
            # self.humanoid located at torch_smpl_humanoid.py
            # self.gt_targets --> keys are (['qpos', 'qvel', 'wbpos', 'wbquat', 'bquat', 'body_com', 'rlinv', 'rlinv_local',
            # 'rangv', 'bangvel', 'ee_wpos', 'ee_pos', 'com', 'height_lb', 'len'])
            gt_targets = self.humanoid[n].qpos_fk(torch.from_numpy(gt_qpos))
            # Initializing target
            target.update({k: data_dict[n][k][:, 0:1, :].clone() for k in self.motion_prior.data_names})

            if self.kin_cfg.model_specs.get("use_tcn", False):
                # this is the HUMOR pose to convert it to world coordinates
                world_body_pos = target['wbpos'].reshape(24, 3)[MUJOCO_2_SMPL][self.smpl_2op_submap]
                if 0:
                    from utils.misc import save_pointcloud
                    inspect_path = f"inspect_out/h_kinres/chi3d_rot/{data_time}/{self.load_context_pass}/"
                    save_pointcloud(world_body_pos, inspect_path + f"world_body_pos.ply")
                    save_pointcloud(target['wbpos'].reshape(24, 3), inspect_path + f"target_wbpos.ply")

                world_trans = world_body_pos[..., 7:8:, :]
                pred_tcn = {
                    'world_body_pos': world_body_pos - world_trans,
                    'world_trans': world_trans,
                }

                casual = self.kin_cfg.model_specs.get("casual_tcn", True)
                full_R, full_t = self.camera_params["full_R"], self.camera_params['full_t']

                if casual: # in the multiphys specs casual_tcn: true
                    joints2d = context["joints2d"][0:1].copy() # shape (1, 12, 3)
                    joints2d[joints2d[..., 2] < self.op_thresh] = 0 # op_thresh=0.1
                    # normalizes joints2d from screen coordinates to unit coordinates
                    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
                    joints2d[..., :2] = normalize_screen_coordinates(joints2d[..., :2], self.camera_params['img_w'],
                                                                     self.camera_params['img_h'])
                    joints2d = np.pad(joints2d, ((self.num_context - 1, 0), (0, 0), (0, 0)), mode="edge")
                else:
                    joints2d = context["joints2d"][:(self.num_context // 2 + 1)].copy()
                    joints2d[joints2d[..., 2] < self.op_thresh] = 0
                    joints2d[..., :2] = normalize_screen_coordinates(joints2d[..., :2], self.camera_params['img_w'],
                                                                     self.camera_params['img_h'])
                    joints2d = np.pad(joints2d, ((self.num_context // 2, self.num_context // 2 + 1 - joints2d.shape[0]),
                                                 (0, 0), (0, 0)), mode="edge")

                # it enters the else
                if self.kin_cfg.model_specs.get("tcn_3dpos", False):
                    world_body_pos = target['wbpos'].reshape(24, 3)[MUJOCO_2_SMPL][self.smpl_2op_submap]
                    world_body_pos = smpl_op_to_op(world_body_pos)
                    cam_body_pos = world_body_pos @ full_R.T + full_t
                    j2d3dfeat = np.concatenate([joints2d[..., :2], np.repeat(cam_body_pos[None,],
                                                                             self.num_context, axis=0)], axis=-1)

                    [self.j2d_seq_feat[n].append(j3dfeat) for j3dfeat in j2d3dfeat]
                    pred_tcn['cam_body_pos'] = cam_body_pos
                else:
                    # at this point the j2dfeat is the normalized 2d joints
                    [self.j2d_seq_feat[n].append(j2dfeat) for j2dfeat in joints2d[..., :2]]

            self.target.append(target)
            self.gt_targets.append(gt_targets)
            self.prev_humor_state.append(prev_humor_state)
            self.cur_humor_state.append(cur_humor_state)
            self.pred_tcn.append(pred_tcn)
        print("***Done loading context")

    def reload_sim_model(self, xml_str):
        """overriding the reload_sim_model from humanoid_im.py"""
        if 0:
            # here, in the first pass of reload_sim_model() the simu doesn't have the scene
            from mujoco_py import MjViewer
            viewer = MjViewer(self.sim)
            viewer.render()
            del viewer
            from pathlib import Path
            def xml_str_to_file(xml_str, filename):
                out_path = Path(filename).parent
                out_path.mkdir(exist_ok=True, parents=True)
                with open(filename, "w") as f:
                    f.write(xml_str)

            filename = "inspect_out/xml_s/humanoid_kin_res_multi/simu.xml"
            xml_str_to_file(xml_str, filename)


        del self.sim
        del self.model
        del self.data
        del self.viewer
        del self._viewers
        # at this point the xml_str contains the two agents
        self.model = mujoco_py.load_model_from_xml(xml_str)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        # distance the agents from each other
        # qpos_np = self.sim.data.qpos[:, None]

        # NUK: this adds distance between the agents, this has no effect
        # self.sim.data.qpos[0] = -0.5
        # self.sim.data.qpos[1] = -0.5
        # self.sim.data.qpos[2] = 0.1

        # self.sim.data.qpos[76] = 0.9
        # self.sim.data.qpos[77] = 0.9
        # self.sim.data.qpos[78] = 0.1

        self.init_qpos = self.sim.data.qpos.copy()
        self.init_qvel = self.sim.data.qvel.copy()
        self.viewer = None
        self._viewers = {}

        if 0:
            self.sim.step()

            from utils.misc import plot
            # self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, offscreen=True, opengl_backend='glfw')
            self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, offscreen=True, opengl_backend='egl')
            # data = self.viewer.read_pixels(width=900, height=900, depth=False)
            # data_v = cv2.flip(data, 0)
            # plot(data_v)
            full_R, full_t = self.camera_params['full_R'], self.camera_params['full_t']
            distance = np.linalg.norm(full_t)
            pos_3d = -full_R.T.dot(full_t)
            # get rotation matrix in euler angles, in degrees
            rotation = sRot.from_matrix(full_R).as_euler("XYZ", degrees=True)
            self.viewer.cam.distance = distance
            lookat = pos_3d  # + x_axis
            self.viewer.cam.lookat[:] = lookat
            self.viewer.cam.azimuth = 90 - rotation[2]  # +10
            self.viewer.cam.elevation = 90 - rotation[0]

            self.viewer.render(width=900, height=900)
            data = self.viewer.read_pixels(width=900, height=900, depth=False)
            data_v = cv2.flip(data, 0)
            plot(data_v)

            # the scene in the simu is visible here, for PROX
            from mujoco_py import MjViewer
            viewer = MjViewer(self.sim)
            viewer.render()

    def set_model_params(self):
        if self.cc_cfg.action_type == 'torque' and hasattr(self.cc_cfg, 'j_stiff'):
            self.model.jnt_stiffness[1:] = self.cc_cfg.j_stiff
            self.model.dof_damping[6:] = self.cc_cfg.j_damp

    def get_obs(self):
        ar_obs = self.get_ar_obs_v1()
        return ar_obs

    def get_cc_obs(self):
        # it enters uhc/envs/humanoid_im.py and uses the function self.get_full_obs_v2()
        # what are here:
        assert self.cc_cfg.obs_type=='full' and self.cc_cfg.obs_v ==2, \
            "ERROR: Should choose another get_full_obs_v from HumanoidEnv - IMPLEMENT!"
        # overwrite get_full_obs_v2 from HumanoidEnv
        # NUK changed by me
        # return super().get_obs()
        return self.get_full_obs_v2()

    def get_full_obs_v2(self, delta_t=0):
        #  change for two people? --> no need for that it can be overwritten in MultiHumanoidKinEnvRes
        #  is this sim data the same as in the MultiHumanoidKinEnvRes env?
        # this inevitably contains info for all agents
        data = self.data
        # self.one_agent_dim
        self.qpos_dim = 76 # is this correct?
        self.qvel_dim = 75
        self.xpos_dim = 24
        qpos_all = data.qpos[:self.qpos_lim].copy() # (152, )
        qvel_all = data.qvel[:self.qvel_lim].copy() # (150, )

        # this reads self.target[n]['qpos']
        target_body_qpos_all = self.get_expert_qpos(delta_t=1 + delta_t)# nos (76,) is that ok?  # target body pose (1, 76)
        target_quat_all = self.get_expert_wbquat(delta_t=1 + delta_t)#.reshape(-1, 4) # (96,)
        target_jpos_all = self.get_expert_joint_pos(delta_t=1 + delta_t) # (72,)
        body_xpos_all = self.data.body_xpos.copy()[1:] # remove world body xpos, left with (48, 3)
        cur_quat_all = self.data.body_xquat.copy()[1:] # remove world body, (48, 4)

        obs_all = []
        for n in range(self.num_agents):
            # n=0 # other mistake!
            qpos_start = self.qpos_dim * n
            qpos_end = self.qpos_dim * (n + 1)
            qvel_start = self.qvel_dim * n
            qvel_end = self.qvel_dim * (n + 1)
            xpos_start = self.xpos_dim * n
            xpos_end = self.xpos_dim * (n + 1)

            qpos = qpos_all[qpos_start:qpos_end].copy() # should be ndarray (76?, )
            qvel = qvel_all[qvel_start:qvel_end].copy() # should be ndarray (75?, )

            # transform velocity
            qvel[:3] = transform_vec(qvel[:3], qpos[3:7], self.cc_cfg.obs_coord).ravel()  # body angular velocity
            obs = []

            qpose_head = qpos[3:7]
            curr_root_quat = self.remove_base_rot(qpose_head)  # shape (4,)
            # todo multi: this is nan for agent 2, why? it gives a diff value than agent 1. Check
            hq = get_heading_q(curr_root_quat)  # is this heading quaternion? shape (4,)
            obs.append(hq)  # obs: heading (4,)
            # ZL : why use heading????. Should remove this...
    
            ######## target body pose #########
            target_body_qpos = target_body_qpos_all[n] # target body pose (1, 76)# it is ok if it's (76,)
            target_quat = target_quat_all[n].reshape(-1, 4) # (24, 4)
            target_jpos = target_jpos_all[n] # (72,)
            ################ Body pose and z ################
            target_root_quat = self.remove_base_rot(target_body_qpos[3:7]) # (4,)
            
            qpos[3:7] = de_heading(curr_root_quat)  # deheading the root, (76,)
            diff_qpos = target_body_qpos.copy()
            diff_qpos[2] -= qpos[2] # compute the difference in z
            diff_qpos[7:] -= qpos[7:] # compute the difference in joint rotations
            diff_qpos[3:7] = quaternion_multiply(target_root_quat, quaternion_inverse(curr_root_quat))
            # obs here gets appended: target qpos, qpos, diff_qpos, but without global transl
            obs.append(target_body_qpos[2:])  # obs: target z + body pose (1, 74)
            obs.append(qpos[2:])  # obs: target z +  body pose (1, 74)
            obs.append(diff_qpos[2:])  # obs:  difference z + body pose (1, 74)
    
            ################ vels ################
            # vel
            # ZL: I think this one has some issues. You are doing this twice.
            qvel[:3] = transform_vec(qvel[:3], curr_root_quat, self.cc_cfg.obs_coord).ravel()
            if self.cc_cfg.obs_vel == "root":
                obs.append(qvel[:6])
            elif self.cc_cfg.obs_vel == "full":
                obs.append(qvel)  # full qvel, 75
    
            ################ relative heading and root position ################
            rel_h = get_heading(target_root_quat) - get_heading(curr_root_quat)
            if rel_h > np.pi:
                rel_h -= 2 * np.pi
            if rel_h < -np.pi:
                rel_h += 2 * np.pi
            # obs: heading difference in angles (1,)
            obs.append(np.array([rel_h])) # (1,) obs: heading difference in root angles
    
            # ZL: this is wrong. Makes no sense. Should be target_root_pos. Should be fixed.
            rel_pos = target_root_quat[:3] - qpos[:3]
            rel_pos = transform_vec(rel_pos, curr_root_quat, self.cc_cfg.obs_coord).ravel()
            obs.append(rel_pos[:2])  # (2,) obs: relative x, y difference (1, 2)
    
            ################ target/difference joint positions ################
            # NUK: this info is for all agents, should be separated --> now it is
            curr_jpos = body_xpos_all[xpos_start:xpos_end] # this is now (24, 3)
    
            # translate to body frame (zero-out root)
            r_jpos = curr_jpos - qpos[None, :3]
            r_jpos = transform_vec_batch(r_jpos, curr_root_quat, self.cc_cfg.obs_coord)  # body frame position
            # obs: target body frame joint position (1, 72)
            obs.append(r_jpos.ravel()) # (72,) obs: target body frame joint position
            diff_jpos = target_jpos.reshape(-1, 3) - curr_jpos
            # print(diff_jpos)
            # print(curr_jpos)
            # print(target_jpos.reshape(-1, 3) )
            # here: diff_jpos is (24, 3), curr_root_quat is (4,), self.cc_cfg.obs_coord='root'
            diff_jpos = transform_vec_batch(diff_jpos, curr_root_quat, self.cc_cfg.obs_coord)
            obs.append(diff_jpos.ravel())  # (72,) obs: current diff body frame joint position  (1, 72)
    
            ################ target/relative global joint quaternions ################
            cur_quat = cur_quat_all[xpos_start:xpos_end]  # think should be (24, 4)
    
            if cur_quat[0, 0] == 0:
                cur_quat = target_quat.copy()
    
            r_quat = cur_quat.copy()
            hq_invert = quaternion_inverse(hq)
            hq_invert_batch = np.repeat(
                hq_invert[None,],
                r_quat.shape[0],
                axis=0,
            )

            # (96,) obs: current target body quaternion (1, 96) # this contains redundant information
            obs.append(quaternion_multiply_batch(hq_invert_batch, r_quat).ravel())
            # (96,) obs: current target body quaternion (1, 96)
            obs.append(quaternion_multiply_batch(quaternion_inverse_batch(cur_quat), target_quat).ravel())

            if self.cc_cfg.has_shape and self.cc_cfg.get("has_shape_obs", True):
                shape_gender_obs = self.get_expert_shape_and_gender()[n]
                obs.append(shape_gender_obs) # (17,) shape_gender_obs
    
            obs = np.concatenate(obs) # is it ok for obs to be (657, )
            obs_all.append(obs)
        # self.cur_t
        # obs_all[1]
        return obs_all

    def get_ar_obs_v1(self):
        t = self.cur_t

        curr_qpos_all = self.data.qpos[:self.qpos_lim].copy()
        curr_qvel = self.data.qvel[:self.qvel_lim].copy()
        self.prev_humor_state = copy.deepcopy(self.cur_humor_state)
        # self.prev_humor_state is a list of 2 dicts one per agent, with keys: (['trans', 'root_orient', 'pose_body', 'joints', 'root_orient_vel', 'joints_vel'])
        #################### MPG is called inside this function ##############
        # MPG is called inside this function
        # proj2dgrad comes from this function
        self.cur_humor_state = humor_dict = self.get_humor_dict_obs_from_sim()
        ######################################################################

        all_obs = []
        qpos_dim_one = 76
        num_agents = self.num_agents

        compute_root_obs = False
        if self.is_root_obs is None:
            self.is_root_obs = []
            compute_root_obs = True
        is_root_obs_all = []
        for n in range(num_agents):
            is_root_obs = []
            obs = []
            # n=0
            start = n * qpos_dim_one
            end = (n + 1) * qpos_dim_one
            curr_qpos = curr_qpos_all[start: end].copy()

            self.pred_joints2d[n].append(humor_dict[n]['pred_joints2d'])
            curr_root_quat = self.remove_base_rot(curr_qpos[3:7]) # (4,)
            full_R, full_t = self.camera_params_torch['full_R'], self.camera_params_torch['full_t']
            target_global_dict = {k: torch.from_numpy(self.context_dict[n][k][(t + 1):(t + 2)].reshape(humor_dict[n][k].shape))
                                  for k in self.motion_prior.data_names}
            conon_output = self.motion_prior.canonicalize_input_double(humor_dict[n], target_global_dict,
                                                                       split_input=False, return_info=True)
            humor_local_dict, next_target_local_dict, info_dict = conon_output
            # print(torch.matmul(humor_dict['trans'], full_R.T) + full_t)
            # info_dict --> keys (['world2aligned_trans', 'world2aligned_rot', 'trans2joint'])
            heading_rot = info_dict['world2aligned_rot'].numpy() # (1, 3, 3)
            curr_body_obs = np.concatenate([humor_local_dict[k].flatten().numpy() for k in self.motion_prior.data_names])
            # curr_body_obs # (336,)
            # hq = get_heading_new(curr_qpos[3:7])
            hq = 0
            obs.append(np.array([hq])) # (1,)
            obs.append(curr_body_obs) # (336,)
            if compute_root_obs:
                is_root_obs.append(np.array([1]))
                is_root_obs.append(np.concatenate([[1 if "root" in k else 0] * humor_local_dict[k].flatten().numpy().shape[-1]
                                                   for k in self.motion_prior.data_names]))

            if self.kin_cfg.model_specs.get("use_tcn", False):
                casual = self.kin_cfg.model_specs.get("casual_tcn", True)
                if casual: # it enters here
                    joints2d_gt = self.context_dict[n]['joints2d'][self.cur_t + 1].copy()
                    joints2d_gt[..., :2] = normalize_screen_coordinates(joints2d_gt[..., :2],
                                                                        self.camera_params['img_w'],
                                                                        self.camera_params['img_h'])
                    joints2d_gt[joints2d_gt[..., 2] < self.op_thresh] = 0
                else:
                    t = self.cur_t + 1
                    pad_num = self.num_context // 2 + 1
                    joints2d_gt = self.context_dict[n]['joints2d'][t:(t + pad_num)].copy()
                    if joints2d_gt.shape[0] < pad_num:
                        joints2d_gt = np.pad(joints2d_gt, ([0, pad_num - joints2d_gt.shape[0]], [0, 0], [0, 0]), mode="edge")

                    joints2d_gt[..., :2] = normalize_screen_coordinates(joints2d_gt[..., :2],
                                                                        self.camera_params['img_w'],
                                                                        self.camera_params['img_h'])
                    joints2d_gt[joints2d_gt[..., 2] < self.op_thresh] = 0 # (12, 3)

                if 0:
                    from utils.misc import plot_joints_cv2
                    black = np.zeros([1080, 1920, 3], dtype=np.uint8)
                    # black = np.zeros([900, 900, 3], dtype=np.uint8)
                    j2d_gt = self.context_dict[n]['joints2d'][self.cur_t + 1].copy()
                    plot_joints_cv2(black, j2d_gt[None], show=True, with_text=True, sc=3)

                if self.kin_cfg.model_specs.get("tcn_3dpos", False):
                    # cam_pred_tcn_3d = humor_dict['cam_pred_tcn_3d']
                    # j2d3dfeat = np.concatenate([joints2d_gt[..., :2], cam_pred_tcn_3d.numpy().squeeze()], axis = 1)
                    cam_pred_3d = humor_dict[n]['cam_pred_3d']
                    cam_pred_3d = smpl_op_to_op(cam_pred_3d)
                    if casual:
                        j2d3dfeat = np.concatenate([joints2d_gt[..., :2], cam_pred_3d.squeeze()], axis=1)
                        self.j2d_seq_feat[n].append(j2d3dfeat)  # push next step obs into state
                    else:
                        j2d3dfeat = np.concatenate([joints2d_gt[..., :2], np.repeat(cam_pred_3d.squeeze(1), self.num_context // 2 + 1, axis=0)], axis=-1)
                        [self.j2d_seq_feat[n].pop() for _ in range(self.num_context // 2)]
                        [self.j2d_seq_feat[n].append(feat) for feat in j2d3dfeat]
                ########################## NUK: it enters here ############################################
                else:
                    if casual: # what is j2d_seq_feat?
                        self.j2d_seq_feat[n].append(joints2d_gt[:, :2])# (12, 2)  # push next step obs into state
                    else:
                        [self.j2d_seq_feat[n].pop() for _ in range(self.num_context // 2)]
                        [self.j2d_seq_feat[n].append(feat) for feat in joints2d_gt[..., :2]]
                ###########################################################################################
                j2d_seq = np.array(self.j2d_seq_feat[n]).flatten() # np.array of (1944,)
                obs.append(j2d_seq) # j2d_seq shape: (1944,) are 81 flattened 12 joints2d, 12*2*81 = 1944
                if compute_root_obs:
                    vari = np.array([3] * j2d_seq.shape[0])
                    is_root_obs.append(vari)

                # use tcn directly on the projection gradient
                tcn_root_grad = self.kin_cfg.model_specs.get("tcn_root_grad", False)# boolean
                world_body_pos, world_trans = self.pred_tcn[n]['world_body_pos'], self.pred_tcn[n]['world_trans'] # (14, 3) and (1, 3)
                curr_body_jts = humor_dict[n]['joints'].reshape(22, 3)[self.smpl_2op_submap].numpy() # (14, 3)
                curr_body_jts -= curr_body_jts[..., 7:8, :] # root relative?
                world_body_pos -= world_body_pos[..., 7:8, :] # ndarray (14, 3)
                body_diff = transform_vec_batch_new(world_body_pos - curr_body_jts, curr_root_quat).T.flatten() # ndarray (42, )
                if 0:
                    from utils.misc import save_pointcloud
                    inspect_path = f"inspect_out/prox/get_ar_obs_v1/"
                    save_pointcloud(world_body_pos, inspect_path + f"world_body_pos_{t:03d}.ply")
                    save_pointcloud(curr_body_jts, inspect_path + f"curr_body_jts_{t:03d}.ply")

                if self.kin_cfg.model_specs.get("tcn_body", False):
                    obs.append(body_diff)
                # todo: target shoudl also be for 2 agents
                curr_trans = self.target[n]['wbpos'][:, :3]   # ndarray (1, 3) # this is in world coord
                trans_diff = np.matmul(world_trans - curr_trans, heading_rot[0].T).flatten()  # ndarray (3,)
                trans_diff[2] = world_trans[:, 2]  # Mimicking the target trans feat.
                if self.kin_cfg.model_specs.get("tcn_traj", False):
                    obs.append(trans_diff)

                if not tcn_root_grad: # enters
                    pred_root_mat = op_to_root_orient(world_body_pos[None,]) #  ndarray (1, 3, 3)
                    root_rot_diff = np.matmul(heading_rot, pred_root_mat).flatten() #  ndarray (9, )
                    obs.append(root_rot_diff) # (9, )

                if self.kin_cfg.model_specs.get("tcn_body", False):
                    if compute_root_obs:
                        is_root_obs.append(np.array([0] * body_diff.shape[0]))

                if self.kin_cfg.model_specs.get("tcn_traj", False):
                    if compute_root_obs:
                        is_root_obs.append(np.array([1] * trans_diff.shape[0]))

                if not tcn_root_grad:
                    if compute_root_obs:
                        is_root_obs.append(np.array([1] * root_rot_diff.shape[0]))

            if self.kin_cfg.model_specs.get("use_rt", True):
                trans_target_local = next_target_local_dict['trans'].flatten().numpy()
                obs.append(trans_target_local)
                if compute_root_obs:
                    is_root_obs.append(np.array([1] * trans_target_local.shape[0]))

            if self.kin_cfg.model_specs.get("use_rr", False):
                root_rot_diff = next_target_local_dict['root_orient'].flatten().numpy()
                obs.append(root_rot_diff)
                if compute_root_obs:
                    is_root_obs.append(np.array([1] * root_rot_diff.shape[0]))

            if self.kin_cfg.model_specs.get("use_3d_grad", False):
                normalize = self.kin_cfg.model_specs.get("normalize_3d_grad", True)
                proj2dgrad = humor_dict[n]['proj2dgrad'].squeeze().numpy().copy()
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
                    is_root_obs.append(np.array([1] * trans_grad.shape[0]))
                obs.append(root_grad)
                if compute_root_obs:
                    is_root_obs.append(np.array([1] * root_grad.shape[0]))
                obs.append(body_grad)
                if compute_root_obs:
                    is_root_obs.append(np.array([1] * body_grad.shape[0]))
            ##### it enters here ##########################################################################################
            elif self.kin_cfg.model_specs.get("use_3d_grad_adpt", False):
                no_grad_body = self.kin_cfg.model_specs.get("no_grad_body", False) # boolean
                proj2dgrad = humor_dict[n]['proj2dgrad'].squeeze().numpy().copy() # ndarray (75, )
                proj2dgrad = np.nan_to_num(proj2dgrad, nan=0, posinf=0, neginf=0)
                proj2dgrad = np.clip(proj2dgrad, -200, 200)
                # sRot if from scipy.spatial.transform rotation
                trans_grad = (np.matmul(heading_rot, proj2dgrad[:3])).squeeze() # ndarray (3, )
                root_grad = (sRot.from_matrix(heading_rot) * sRot.from_rotvec(proj2dgrad[3:6])).as_rotvec().squeeze() # ndarray (3, )
                body_grad = proj2dgrad[6:69] # ndarray (63, )
                if no_grad_body:
                    # Ablation, zero body grad. Just TCN
                    body_grad = np.zeros_like(body_grad)
                obs.append(trans_grad) # (3,)
                if compute_root_obs:
                    is_root_obs.append(np.array([1] * trans_grad.shape[0]))
                obs.append(root_grad) # (3,)
                if compute_root_obs:
                    is_root_obs.append(np.array([1] * root_grad.shape[0]))
                obs.append(body_grad) # (63,)
                if compute_root_obs:
                    is_root_obs.append(np.array([1] * body_grad.shape[0]))
            ################################################################################################################

            if self.kin_cfg.model_specs.get("use_sdf", False):
                sdf_vals = self.smpl_robot[n].get_sdf_np(self.cur_humor_state[n]['joints'].reshape(-1, 3), topk=3)
                obs.append(sdf_vals.numpy().flatten())
                if compute_root_obs:
                    is_root_obs.append(np.array([2] * sdf_vals.shape[0]))
            elif self.kin_cfg.model_specs.get("use_dir_sdf", False):
                sdf_vals, sdf_dirs = self.smpl_robot[n].get_sdf_np(self.cur_humor_state[n]['joints'].reshape(-1, 3), topk=3, return_grad=True)
                sdf_dirs = np.matmul(sdf_dirs, heading_rot[0].T)  # needs to be local dir coord
                sdf_feat = (sdf_vals[:, :, None] * sdf_dirs).numpy().flatten()
                obs.append(sdf_feat)
                if compute_root_obs:
                    is_root_obs.append(np.array([2] * sdf_feat.shape[0]))
            ################ VOXEL observations ############################################################################
            ########################### it enters here #####################################################################
            if self.kin_cfg.model_specs.get("use_voxel", False):
                voxel_res = self.kin_cfg.model_specs.get("voxel_res", 8) # this is =16
                # these voxel_feat are float continuous values
                voxel_feat = self.smpl_robot[n].query_voxel(self.cur_humor_state[n]['trans'].reshape(-1, 3),
                                                         self.cur_humor_state[n]['root_orient'].reshape(3, 3),
                                                         res=voxel_res).flatten() # (4096,)
                # these are booleans of shape (4096,) and self.voxel_thresh=0.1
                inside, outside = voxel_feat <= 0, voxel_feat >= self.voxel_thresh

                if 0:
                    from skimage import measure
                    import trimesh
                    inside_ = inside.reshape(voxel_res, voxel_res, voxel_res)
                    plot_voxel(inside_)
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
                    is_root_obs.append(np.array([2] * voxel_feat.shape[0]))
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
                    is_root_obs.append(np.array([0] * contact_feat.shape[0]))
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
                is_root_obs = np.concatenate(is_root_obs)
                assert (is_root_obs.shape == obs.shape)

            all_obs.append(obs)
            is_root_obs_all.append(is_root_obs)
        # this should be both elements of (6455,)
        self.is_root_obs = is_root_obs_all

        return all_obs

    def step_ar(self, action_all, dt=1 / 30):
        # action is (114,)
        cfg = self.kin_cfg
        self.target = [[] for i in range(self.num_agents)] # overwrites previous target values

        for n in range(self.num_agents):
            action = action_all[n].copy()
            # change this (action) to number of joints
            # cur_humor_state --> (['trans_vel', 'joints_vel', 'root_orient_vel', 'joints',
            # 'pose_body', 'root_orient', 'trans', 'pred_joints2d', 'cam_pred_3d', 'proj2dgrad'])
            action_ = torch.from_numpy(action[None, :69])
    
            #### motion_prior.step_state() ####
            # this actually ONLY does some transformations and
            # applies the predicted residuals from the kinematic policy to the pose
            next_global_out = self.motion_prior.step_state(self.cur_humor_state[n], action_)
            # next_global_out --> includes (['trans', 'root_orient', 'pose_body'])

            # This specifies the TARGE pose
            body_pose_aa = mat2aa(next_global_out['pose_body'].reshape(21, 3, 3)).reshape(1, 63)
            root_aa = mat2aa(next_global_out['root_orient'].reshape(1, 3, 3)).reshape(1, 3)
            pose_aa = torch.cat([root_aa, body_pose_aa, torch.zeros(1, 6).to(root_aa)], dim=1)  # (1, 72)

            # get the GT data from self.context_dict, possibly 'pose_body'
            overwrite_target_w_gt = self.kin_cfg.overwrite_target_w_gt
            if overwrite_target_w_gt:
                if self.cur_t == 0:
                    print("*** Warning: OVERRIDING TARGET WITH GT ***")
                gt_pose = torch.from_numpy(self.context_dict[n]['pose_body'].reshape(-1, 3, 3))
                body_pose_aa_gt = mat2aa(gt_pose).reshape(-1, 63)
                N = body_pose_aa_gt.shape[0]
                gt_root = torch.from_numpy(self.context_dict[n]['root_orient'].reshape(-1, 3, 3))
                root_aa_gt = mat2aa(gt_root).reshape(-1, 3)
                pose_aa_gt = torch.cat([root_aa_gt, body_pose_aa_gt, torch.zeros(N, 6).to(root_aa)], dim=1)  # (1, 72)
                trans_gt = torch.from_numpy(self.context_dict[n]["trans"])
                # self.cur_t
                pose_aa = pose_aa_gt[self.cur_t].reshape(1, -1)
                next_global_out['trans'] = trans_gt[self.cur_t].reshape(1, 1, -1)
                next_global_out['root_orient'] = gt_root[self.cur_t].reshape(1, 1, -1)
                gt_pose_ = gt_pose.reshape(-1, 21, 3, 3).reshape(-1, 21*3*3)
                next_global_out['pose_body'] = gt_pose_[self.cur_t].reshape(1, 1, -1)
            
            # multi: this should be multi-agent
            # qpos = smpl_to_qpose_torch(pose_aa, self.model, trans=next_global_out['trans'].reshape(1, 3),
            #                            count_offset=True) # (1, 76)
            qpos = smpl_to_qpose_torch_multi(pose_aa, self.model, trans=next_global_out['trans'].reshape(1, 3),
                                       count_offset=True, agent_id=n) # (1, 76)

            # HUMANOID is used here, HUMANOID needs to be multi-agent
            # multi: make use of humanoid multi-agent
            if self.mode == "train" and self.agent.iter < self.agent.num_supervised and self.agent.iter >= 0:
                # Dagger
                qpos = torch.from_numpy(self.gt_targets[n]['qpos'][(self.cur_t):(self.cur_t + 1)])
                fk_res = self.humanoid[n].qpos_fk(qpos)
            else:
                # fk_res --> keys:
                # (['qpos', 'qvel', 'wbpos', 'wbquat', 'bquat', 'body_com', 'rlinv', 'rlinv_local',
                # 'rangv', 'bangvel', 'ee_wpos', 'ee_pos', 'com', 'height_lb', 'len'])
                #####################################################################################
                # qpos is different for agent 2 when I use it as the first agent (works) vs. when
                # used as second agent (doesn't work)
                fk_res = self.humanoid[n].qpos_fk(qpos) # dict of 15 elements

            if 0:
                from utils.smpl import smpl_to_verts
                from utils.smpl import from_qpos_to_smpl
                from utils.misc import save_trimesh
                from utils.misc import save_pointcloud
                t = self.cur_t
                trans = next_global_out["trans"]
                inspect_path = f"inspect_out/step_ar/chi3d/"
                next_prior_verts, faces = smpl_to_verts(pose_aa, trans[0])
                # pose_aa --> is correct, meaning self.motion_prior.step_state() is correct
                save_trimesh(next_prior_verts[0, 0], faces, inspect_path + f"pose_aa_{n}.ply")
                # qpos --> converted to smpl is correct
                qpos_v, faces = from_qpos_to_smpl(qpos[0], self) # input has to be (76,)
                save_trimesh(qpos_v[0, 0], faces, inspect_path + f"qpos_{n}.ply")
                # wbpos --> it is correct, these are joints, not smpl params!
                wbpos = fk_res['wbpos']
                save_pointcloud(wbpos[0].reshape(-1, 3), inspect_path + f"wbpos_{n}.ply")

                for t, pose in enumerate(pose_aa_gt):
                    target_gt_v, faces = smpl_to_verts(pose[None], trans_gt[t, None])
                    save_trimesh(target_gt_v[0, 0], faces, inspect_path + f"gts/{n:02d}/gt_pose_{t:02d}.ply")


            # todo multi bug: overwrite this with GT and see if output is correct
            # wbpos, print(fk_res['wbpos'].reshape(-1, 3))
            self.target[n] = fk_res
            # updates the target with the next pose dictated by the kinematic policy and not actually HUMOR
            # all the quantities here are for one time step or one frame only
            # next_global_out actually has only keys: ['trans', 'root_orient', 'pose_body'],
            # this contains the new target pose predicted by the kinematic policy
            self.target[n].update(next_global_out)  # updates to dict of 18 elements
            if self.kin_cfg.model_specs.get("use_tcn", False):
                full_R, full_t = self.camera_params['full_R'], self.camera_params['full_t']
                kp_feats = action[69:].copy()
                cam_trans = kp_feats[None, :3]
                cam_body_pos = kp_feats[3:].reshape(14, 3)
                # camera to world transformation
                self.pred_tcn[n]['world_trans'] = (cam_trans - full_t).dot(full_R)
                # world_body_pos are the (prev_step?) joints from pose_aa or next_global_out (which is the same)
                # it comes from action
                world_body_pos = cam_body_pos.dot(full_R)
                self.pred_tcn[n]['world_body_pos'] = world_body_pos
                self.pred_tcn[n]['cam_body_pos'] =  cam_trans + cam_body_pos
    
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
                pred_joints2d = self.cur_humor_state[n]["pred_joints2d"]
                # this pred_joints2d has very weird values for CHI3D data
                pred_joints2d_ = pred_joints2d[0, 0].cpu().numpy()



    def get_humanoid_pose_aa_trans(self, qpos=None, agent_id=None):
        assert agent_id is not None, "agent_id must be specified"
        if qpos is None:
            qpos = self.data.qpos.copy()[None]
        # pose_aa, trans = qpos_to_smpl(qpos, self.model, self.cc_cfg.robot_cfg.get("model", "smpl"))
        pose_aa, trans = qpos_to_smpl_multi(qpos, self.model, self.cc_cfg.robot_cfg.get("model", "smpl"), agent_id)
        # qpos is (2, 76), pose_aa is (2, 24, 3), trans is (2, 3)
        return pose_aa, trans

    def get_humor_dict_obs_from_sim(self):
        """ NUK: Compute obs based on current and previous simulation state and coverts it into humor format. """
        # gets both the current and previous qpos
        # self.data is the mujoco data, so this comes from mujoco loaded simulation
        qpos_dim_one = 76
        num_agents = self.num_agents
        qpos_all = self.data.qpos.copy()[None] # (1, 76) but with agent2 qpos is (1, 152)
        # qpos = self.get_expert_qpos()[None] # No simulate
        prev_qpos_all = self.prev_qpos[None] # (1, 76)

        all_humor = []
        for n in range(num_agents):
            start = n * qpos_dim_one
            end = (n + 1) * qpos_dim_one
            qpos = qpos_all[:, start: end] # (1, 76)
            prev_qpos = prev_qpos_all[:, start: end] # (1, 76)

            # # NUK hack for now
            # qpos = qpos[:, :76]
            # prev_qpos = prev_qpos[:, :76]

            # Calculating the velocity difference from simulation. We do not use target velocity.
            qpos_stack = np.concatenate([prev_qpos, qpos])
            # No need to be multi inside get_humanoid_pose_aa_trans
            pose_aa, trans = self.get_humanoid_pose_aa_trans(qpos_stack, agent_id=n) # Simulation state.

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
            # qpos_fk depends on the betas, in that case, I need to have two
            fk_result = self.humanoid[n].qpos_fk(torch.from_numpy(qpos_stack), to_numpy=False, full_return=False)
            trans_batch = torch.from_numpy(trans[None]) # ([1, 2, 3])

            joints = fk_result["wbpos"].reshape(-1, 24, 3)[:, MUJOCO_2_SMPL].reshape(-1, 72)[:, :66] # (2,, 66)
            pose_aa_mat = aa2mat(torch.from_numpy(pose_aa.reshape(-1, 3))).reshape(1, 2, 24, 4, 4)[..., :3, :3] # ([1, 2, 24, 3, 3])
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
            joints2d_gt = self.context_dict[n]['joints2d'][t:(t + grad_frame_num)].copy() # (1, 12, 3)

            if joints2d_gt.shape[0] < grad_frame_num:
                try:
                    joints2d_gt = np.pad(joints2d_gt, ([0, grad_frame_num - joints2d_gt.shape[0]], [0, 0], [0, 0]), mode="edge")
                except:
                    print('bug!')
            inliers = joints2d_gt[..., 2] > self.op_thresh # boolean: (1, 12)
            self.hb[n].update_tgt_joints(joints2d_gt[..., :2], inliers)
            if 0:
                from utils.misc import read_image_PIL, plot_joints_cv2
                seq_name = self.context_dict[n]['seq_name']
                img_path = f"/sailhome/nugrinov/code/CVPR_2024/slahmr_release/slahmr/videos/viw/images/{seq_name}"
                img = read_image_PIL(f"{img_path}/{t:06d}.jpg")
                plot_joints_cv2(img, joints2d_gt, show=True, with_text=True, sc=3)
                feet_i = [9, 11]

            # input_vect contains the SMPL pose corresponding to current qpos only
            input_vec = np.concatenate([humor_out['trans'].numpy(), pose_aa[1:2].reshape(1, -1, 72)], axis=2) # (1, 1, 75)

            ######################################## Projection of 3D to 2D keypoints ######################################
            # pred_2d --> (1, 12, 2)
            # cam_pred_3d --> (1, 14, 2)
            # self.hb.proj2d --> projects 3D keypoints to 2D using the camera parameters and SMPL joints format
            data_name = self.kin_cfg.data_name
            pred_2d, cam_pred_3d = self.hb[n].proj2d(fk_result["wbpos"][1:2].reshape(24, 3).numpy(), return_cam_3d=True,
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
                pose_grad, input_vec_new, curr_loss = self.multi_step_grad(input_vec, order=order, num_adpt_grad=num_adpt_grad,
                                                                           normalize=normalize, step_size=grad_step, agent_id=n)
                multi = depth / 10
                pose_grad[:6] *= multi
                humor_out["proj2dgrad"] = pose_grad

            elif self.kin_cfg.model_specs.get("use_3d_grad_line", False):
                proj2dgrad = self.proj_2d_line_loss[n](input_vec)
                humor_out["proj2dgrad"] = -torch.from_numpy(proj2dgrad)

            elif self.kin_cfg.model_specs.get("use_3d_grad_adpt", False):
                num_adpt_grad = self.kin_cfg.model_specs.get("use_3d_grad_adpt_num", 5)
                grad_step = self.kin_cfg.model_specs.get("grad_step", 5)

                ############################################################################################################
                ######################### This is the MPG !!! ##############################################################
                # todo multi: does this needs to be done per agent changing self.hb?
                pose_grad, input_vec_new, curr_loss = self.multi_step_grad(input_vec,
                                                                           order=order,
                                                                           num_adpt_grad=num_adpt_grad,
                                                                           normalize=normalize,
                                                                           step_size=grad_step,
                                                                           agent_id=n
                                                                           )
                ############################################################################################################
                # I guess pose_grad follows the same other as input_vec --> (transl, pose_aa) or (transl, smpl_pose_72)
                multi = depth / 10
                pose_grad[:6] *= multi
                humor_out["proj2dgrad"] = pose_grad # (1, 1, 75)
                # also defined here:
                # humor_out["pred_joints2d"]
                # humor_out["cam_pred_3d"]
                if 0:
                    from utils.misc import plot_joints_cv2
                    # black = np.zeros([1080, 1920, 3], dtype=np.uint8)
                    black = np.zeros([900, 900, 3], dtype=np.uint8)
                    pred_joints2d = humor_out["pred_joints2d"]
                    plot_joints_cv2(black, pred_joints2d[0], show=True, with_text=True, sc=3)

            all_humor.append(humor_out)
            # todo: joints 2d are only slightly different from each other, this should not be. check!
            # pred_joints2d_1 = all_humor[0]['pred_joints2d']
            # pred_joints2d_2 = all_humor[1]['pred_joints2d']

        return all_humor

    def geo_trans(self, input_vec_new, agent_id=None):
        n = agent_id
        assert agent_id is not None, "agent_id is None, but it should be an integer"
        delta_t = np.zeros(3)
        geo_tran_cap = self.kin_cfg.model_specs.get("geo_trans_cap", 0.1)
        try:
            inliners = self.hb[n].inliers
            if np.sum(inliners) >= 3:
                wbpos = self.hb[n].fk_batch_grad(input_vec_new)
                cam2d, cam3d = self.hb[n].proj2d(wbpos, True)
                cam3d = smpl_op_to_op(cam3d)
                j2ds = self.hb[n].gt_2d_joints[0].copy()
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
                delta_t = self.hb[n].full_R.T @ delta_t[:, 0]

                if np.linalg.norm(delta_t) > geo_tran_cap:
                    delta_t = delta_t / np.linalg.norm(delta_t) * geo_tran_cap
            else:
                delta_t = np.zeros(3)

        except Exception as e:
            print("error in svd and pose", e)
        return delta_t


    def multi_step_grad(self, input_vec, num_adpt_grad=5, normalize=False, order=2, step_size=5, agent_id=None):
        assert agent_id is not None, "agent_id is None, needs to be set"
        geo_trans = self.kin_cfg.model_specs.get("geo_trans", False)
        tcn_root_grad = self.kin_cfg.model_specs.get("tcn_root_grad", False)
        input_vec_new = input_vec.copy()
        #########
        # this loss is important, it generates the 2D keypoints from the input_vec by projecting them into the img,
        # using cam params
        prev_loss = orig_loss = self.hb[agent_id].proj_2d_loss(input_vec_new, ord=order, normalize=normalize)
        #############
        if tcn_root_grad: # does not enter here
            world_body_pos, world_trans = self.pred_tcn['world_body_pos'], self.pred_tcn['world_trans']
            pred_root_vec = sRot.from_matrix(op_to_root_orient(world_body_pos[None,])).as_rotvec()  # tcn's root
            input_vec_new[..., 3:6] = pred_root_vec

        if order == 1:
            step_size = 0.00001
            step_size_a = step_size * np.clip(prev_loss, 0, 5)
        else:
            if normalize: # does not normalize
                step_size_a = step_size / 1.02
            else: # it enters here
                step_size_a = 0.000005
        for iteration in range(num_adpt_grad): # num_adpt_grad=5
            # it enters the if
            if self.kin_cfg.model_specs.get("use_3d_grad_sept", False): #enters here
                proj2dgrad_body = self.proj_2d_body_loss[agent_id](input_vec_new, ord=order, normalize=normalize)
                proj2dgrad = self.proj_2d_loss[agent_id](input_vec_new, ord=order, normalize=normalize)
                proj2dgrad[..., 3:] = proj2dgrad_body[..., 3:]
                proj2dgrad = np.nan_to_num(proj2dgrad, posinf=0, neginf=0)  # This is essentail, otherwise nan will get more
            else:
                proj2dgrad = self.proj_2d_loss[agent_id](input_vec_new, ord=order, normalize=normalize)
                proj2dgrad = np.nan_to_num(proj2dgrad, posinf=0, neginf=0)  # This is essentail, otherwise nan will get more

            # import ipdb
            # ipdb.set_trace()
            # wbpos = self.hb.fk_batch_grad(input_vec_new); pred_joints2d = self.hb.proj2d(wbpos); joblib.dump(pred_joints2d, "a.pkl"); joblib.dump(self.hb.gt_2d_joints, "b.pkl")

            input_vec_new = input_vec_new - proj2dgrad * step_size_a

            if geo_trans:
                delta_t = self.geo_trans(input_vec_new, agent_id=agent_id)
                delta_t = np.concatenate([delta_t, np.zeros(72)])
                input_vec_new += delta_t

            curr_loss = self.hb[agent_id].proj_2d_loss(input_vec_new, ord=order, normalize=normalize)

            if curr_loss > prev_loss:
                step_size_a *= 0.5
            prev_loss = curr_loss

        if self.hb[agent_id].proj_2d_loss(input_vec_new, ord=order, normalize=normalize) > orig_loss:
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
        # a here is action and comes from the kinematic policy, contains
        # information for updating the kinematic target pose --> shape(114, )
        fail = False
        cfg = self.kin_cfg
        cc_cfg = self.cc_cfg

        self.prev_qpos = self.get_humanoid_qpos() # now (152, )
        # self.prev_qpos = self.get_expert_qpos() ## No simulate
        # print(self.prev_qpos[76:])
        self.prev_qvel = self.get_humanoid_qvel() # now (150, )
        self.prev_bquat = self.bquat.copy() # now (188, ) ??
        self.prev_hpos = self.get_head().copy() # now (7, ) ??

        ########################## important function ##########################
        # here the kinematic target pose is updated with the residuals that come
        # from the kinematic policy.
        # this step_ar is related to self.target['wbpos'],
        # here it sets self.target = fk_res
        # fk_res --> is the kinematic target that the UHC will try to mimic
        # does a.copy() inside
        # a is action from the kin-poly (not action for the UHC)
        self.step_ar(a)
        ########################################################################

        # if flags.debug:
        # self.target = self.humanoid.qpos_fk(torch.from_numpy(self.context_dict['qpos'][self.cur_t:self.cur_t + 1])) # GT
        # self.target = self.smpl_humanoid.qpos_fk(self.ar-*m_numpy(self.gt_targets['qpos'][self.cur_t:self.cur_t + 1])) # GT
        # self.target = self.humanoid.qpos_fk(torch.from_numpy(self.gt_targets['qpos'][self.cur_t:self.cur_t + 1])) # Use gt

        # do N inters without updating the target kinematic pose, a sort of "inner loop"
        for inner_iters in range(self.kin_cfg.loops_uhc):
            ####################################### UHC step ##############################################
            # gets the obs that DOES NOT contain scene voxels
            cc_a_all = []
            # calls self.get_full_obs_v2(), for multi-agent it overrides super().get_obs()
            cc_obs_all = self.get_cc_obs() # runs super().get_obs() # cc_obs is (657, )??
            # this will loop over each agent, where n is the agent id
            for n, cc_obs in enumerate(cc_obs_all):
                # this runs ZFilter() from uhc/khrylib/utils/zfilter.py --> does y = (x-mean)/std
                cc_obs = self.cc_running_state(cc_obs, update=False)
                ########################################### CC step ##########################################
                # it is the PolicyGaussian, so it is the UHC policy
                # NUK: as cc_a is the input to the simulation
                # cc_a is (315, ) same dim as the action space
                # this calls select_action() from class Policy() --> calls forward() def. in class PolicyGaussian()
                cc_a = self.cc_policy.select_action(torch.from_numpy(cc_obs)[None,], mean_action=True)[0].numpy()
                cc_a_all.append(cc_a)
                ##############################################################################################

            ################################ Physical simulation occurs here ##############################################
            # do not concat
            # cc_a_all = np.concatenate(cc_a_all) # cc_a_all is (630, )

            if flags.debug:
                self.do_simulation(cc_a_all, self.frame_skip)
                # self.data.qpos[:self.qpos_lim] = self.gt_targets['qpos'][self.cur_t + 1]  # debug
                # self.sim.forward()  # debug
            else:
                ###### normal operation
                if kin_override:
                    # debug
                    gt_qposes = [self.gt_targets[i]['qpos'][self.cur_t + 1] for i in range(self.num_agents)]
                    self.data.qpos[:self.qpos_lim] = np.concatenate(gt_qposes)
                    self.sim.forward()  # debug
                else:
                    # it comes here by default
                    if self.simulate:
                        try:
                            # print(self.data.qpos[76:]) # at this point it is all the same in normal op
                            # goes to HumanoidEnv--> now overwriten in MultiHumanoidKinEnvRes
                            self.do_simulation(cc_a_all, self.frame_skip)
                        except Exception as e:
                            print("Exception in do_simulation", e, self.cur_t)
                            fail = True
                    else:
                        # debug
                        expert_qpos = self.get_expert_qpos()
                        self.data.qpos[:self.qpos_lim] = np.concatenate(expert_qpos)
                        self.sim.forward()  # debug

            pass
            if 0:
                from utils.smpl import smpl_to_verts
                from utils.smpl import from_qpos_to_smpl
                from utils.misc import save_trimesh
                from utils.misc import save_pointcloud
                t = self.cur_t
                for n in range(2):
                    # n = 0
                    target = self.target[n]
                    qpos = target['qpos']
                    trans = target["trans"]
                    inspect_path = f"inspect_out/two_agent/step/chi3d/"
                    # qpos --> converted to smpl is correct
                    qpos_v, faces = from_qpos_to_smpl(qpos[0], self) # input has to be (76,)
                    save_trimesh(qpos_v[0, 0], faces, inspect_path + f"{t:02d}/qpos_{n}.ply")
                    # wbpos --> it is correct, these are joints, not smpl params!
                    simu_qpos = self.data.qpos[n*self.qpos_dim:(n+1)*self.qpos_dim]
                    qpos_v, faces = from_qpos_to_smpl(simu_qpos, self) # input has to be (76,)
                    save_trimesh(qpos_v[0, 0], faces, inspect_path + f"{t:02d}/simu_qpos_{n}.ply")
                #
            # if self.cur_t == 0 and self.agent.global_start_fr == 0:
            #     # ZL: Stablizing the first frame jump
            #     self.data.qpos[:self.qpos_lim] = self.get_expert_qpos()  # debug
            #     self.data.qvel[:] = 0
            #     self.sim.forward()  # debug
            ##############################################################################################
            self.bquat = self.get_body_quat()

        # put this outside, to not change the iteration count!
        self.cur_t += 1
        # get obs
        reward = 1.0
        if cfg.env_term_body == 'body':
            # ignoring for now. todo multi: change
            # body_diff = self.calc_body_diff()
            body_diff = 0
            if self.mode == "train":
                body_gt_diff = self.calc_body_gt_diff()
                fail = fail or (body_diff > 2.5 or body_gt_diff > 3.5)
            else:
                if cfg.ignore_fail:
                    fail = False
                else:
                    fail = fail or body_diff > 7
        else:
            raise NotImplemented()


        end = (self.cur_t >= cc_cfg.env_episode_len) or (self.cur_t + self.start_ind >= self.context_dict[0]['len'])
        done = fail or end
        percent = self.cur_t / self.context_dict[0]['len']

        ############################## The new observation is computed here #########################################
        if not done:
            # NUK: this one calls self.get_ar_obs_v1() and it is different from self.get_cc_obs() which calls
            # self.get_full_obs_v2()()
            # NOTE: MPG is called inside this get_obs() function
            # NOTE: now obs is a list of len 2
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

    def set_state(self, qpos, qvel):
        """ NUK: This function overrides the MujocoEnv function"""

        print('**** Setting state for 2 AGENTS ****')
        qpose_comb = np.concatenate(qpos)
        qvel_comb = np.concatenate(qvel)

        # self.model.nq is 152 for two people, 76 for one person
        # self.model.nv is 150 for two people, 75 for one person
        # when called from reload_sim_model():
        # qpos is 228, qvel is 225 and self.model.nq and self.model.nv are the same, respectively. Why?
        assert qpose_comb.shape == (self.model.nq,) and qvel_comb.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        # acordding to the documentation, MjSimState:
        # Represents a snapshot of the simulator’s state. This includes time, qpos, qvel, act, and udd_state.
        # so this first gets the states of the current simulation and then with this, it updates the sim?
        new_state = mujoco_py.MjSimState(
            old_state.time, qpose_comb, qvel_comb, old_state.act, old_state.udd_state
        )
        # print(qpose_comb[76:])
        # here it cleary sets the simu to a new state, but why get it from MjSimState?
        self.sim.set_state(new_state)
        # self.data.qpos[76:], self.data.qpos[:76]
        self.sim.forward()

    def reset_model(self, qpos=None, qvel=None):
        cfg = self.kin_cfg
        ind = 0
        self.start_ind = 0 # what is this???

        self.qpos_dim = 76 # is this correct?
        self.qvel_dim = 75
        self.xpos_dim = 24

        if qpos is None:
            init_qpos_all = []
            init_vel_all = []
            for n, context_dict in enumerate(self.context_dict):
                init_pose_aa = context_dict['init_pose_aa']
                init_trans = context_dict['init_trans']
                # init_qpos = smpl_to_qpose(torch.from_numpy(init_pose_aa[None,]),
                #                           self.model, torch.from_numpy(init_trans[None,]),
                #                           count_offset=True).squeeze()
                # args should be in order: pose, model, trans
                init_qpos = smpl_to_qpose_multi(torch.from_numpy(init_pose_aa[None,]),
                                                self.model, torch.from_numpy(init_trans[None,]),
                                                count_offset=True, agent_id=n).squeeze()
                # self.data.qpos
                # print(init_qpos)
                init_vel = np.zeros(self.qvel_dim)

                init_qpos_all.append(init_qpos) # init_qpos is (76, ) ?
                init_vel_all.append(init_vel) # init_qpos is (75, ) ?

        else:
            # init_qpos = qpos
            # init_vel = qvel
            init_qpos_all = qpos
            init_vel_all = qvel

        #######################
        # I think that here the pose is updated to the init pose
        if 0:
            sim = mujoco_py.MjSim(self.model)
            from mujoco_py import MjViewer
            viewer = MjViewer(sim)
            viewer.render()
        # sets initial poses to start the simu
        self.set_state(init_qpos_all, init_vel_all)
        # print(self.data.qpos[76:]),
        # print(self.data.qpos[:76])
        if 0:
            sim.step()
            viewer.render()

        #######################
        self.prev_qpos = self.get_humanoid_qpos() # 152, and 152/2 = 76

        ################################### GET OBS #################################
        obs = self.get_obs()
        #############################################################################

        if 0:
            sim.step()
            viewer.render()
        # obs is now a list of len 2, each with obs of (6455,)
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

    def get_expert_kin_pose(self, delta_t=0):
        expert_qpos_all = self.get_expert_qpos(delta_t=delta_t)
        out_all = []
        for n in range(self.num_agents):
            # each expert_qpos is (76, )
            # out is (69, ), this matches the number of actuators for each robot
            out = expert_qpos_all[n][7:]
            out_all.append(out)
        # return self.get_expert_qpos(delta_t=delta_t)[7:]
        return out_all

    def get_expert_qpos(self, delta_t=0):
        # expert_qpos = self.target["qpos"].copy()
        expert_qpos_all = []
        for n in range(self.num_agents):
            expert_qpos = self.target[n]['qpos'].copy().squeeze()
            expert_qpos_all.append(expert_qpos)
        return expert_qpos_all

    def get_target_kin_pose(self, delta_t=0): # todo this is not done yet
        target_kin_pose_all = []
        for n in range(self.num_agents):
            target_kin_pose = self.get_expert_qpos()[7:]
            target_kin_pose_all.append(target_kin_pose)
        return target_kin_pose_all

    def get_expert_joint_pos(self, delta_t=0):
        # world joint position
        # wbpos = self.target['wbpos'].squeeze()
        wbpos_all = []
        for n in range(self.num_agents):
            wbpos = self.target[n]['wbpos'].squeeze()
            wbpos_all.append(wbpos)
        return wbpos_all

    def get_expert_com_pos(self, delta_t=0):
        # body joint position
        # body_com = self.target['body_com'].squeeze()
        body_com_all = []
        for n in range(self.num_agents):
            body_com = self.target[n]['body_com'].squeeze()
            body_com_all.append(body_com)
        return body_com_all

    def get_expert_bquat(self, delta_t=0):
        # bquat = self.target['bquat'].squeeze()
        bquat_all = []
        for n in range(self.num_agents):
            bquat = self.target[n]['bquat'].squeeze()
            bquat_all.append(bquat)
        return bquat_all

    def get_expert_wbquat(self, delta_t=0):
        # wbquat = self.target['wbquat'].squeeze()
        wbquat_all = []
        for n in range(self.num_agents):
            wbquat = self.target[n]['wbquat'].squeeze() # (96,)
            wbquat_all.append(wbquat)
        return wbquat_all

    def get_expert_shape_and_gender(self):
        cfg = self.cc_cfg
        obs_all = []
        for n in range(self.num_agents):
            shape = self.context_dict[n]['beta'][0].squeeze()
            if shape.shape[0] == 10:
                shape = np.concatenate([shape, np.zeros(6)])

            gender = self.context_dict[n]['gender'][0].squeeze()
            obs = []
            if cfg.get("has_pca", True):
                obs.append(shape)

            obs.append([gender])

            if cfg.get("has_weight", False):
                # it seems that it never uses this weight
                obs.append([self.weight])

            if cfg.get("has_bone_length", False):
                obs.append(self.smpl_robot.bone_length)
            obs = np.concatenate(obs)
            obs_all.append(obs)
        return obs_all

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

    def compute_desired_accel(self, qpos_err, qvel_err, k_p, k_d):
        """ NUK: this is multi agent version of the original compute_torque from HumanoidEnv"""
        dt = self.model.opt.timestep
        nv = self.model.nv

        M = np.zeros(nv * nv)
        # This function multiplies the joint-space inertia matrix stored in mjData.qM by a vector.
        # qM has a custom sparse format that the user should not attempt to manipulate directly.
        # Alternatively one can convert qM to a dense matrix with mj_fullM and then user regular
        # matrix-vector multiplication, but this is slower because it no longer benefits from sparsity.
        # todo multi: this could also be a potential source of errors, understand what this does
        mjf.mj_fullM(self.model, M, self.data.qM)
        M.resize(nv, nv)
        M = M[:self.qvel_lim, :self.qvel_lim]
        C = self.data.qfrc_bias.copy()[:self.qvel_lim]
        K_p = np.diag(k_p)
        K_d = np.diag(k_d)
        q_accel = cho_solve(
            cho_factor(M + K_d * dt, overwrite_a=True, check_finite=False),
            -C[:, None] - K_p.dot(qpos_err[:, None]) - K_d.dot(qvel_err[:, None]),
            overwrite_b=True,
            check_finite=False,
        )
        return q_accel.squeeze()

    def compute_torque(self, ctrl_all, i_iter=0): #  i_iter=number of skip frames
        """ NUK: this is multi agent version of the original compute_torque from HumanoidEnv
        ctrl_all: are the actions that come from the UHC policy
        """
        cfg = self.cc_cfg
        dt = self.model.opt.timestep
        qpos_all = self.get_humanoid_qpos() # (152, )
        qvel_all = self.get_humanoid_qvel() # (150, )

        # multi: made multi agent version
        if self.cc_cfg.action_v == 1:
            # should use the target pose instead of the current pose
            base_pos_all = self.get_expert_kin_pose(delta_t=1)
        elif self.cc_cfg.action_v == 0:
            base_pos_all = cfg.a_ref
            raise NotImplementedError
        else:
            raise NotImplementedError

        qpos_err_all = []
        qvel_err_all = []
        k_p_all = []
        k_d_all = []
        curr_jkp_all = []
        curr_jkd_all = []
        for n in range(self.num_agents):
            # n=0 # --> mistake
            ctrl = ctrl_all[n] # (315, ) where does this come from?
            ctrl_joint = ctrl[:self.ndof]

            qpos = qpos_all[n * self.qpos_dim : (n + 1) * self.qpos_dim]
            qvel = qvel_all[n * self.qvel_dim : (n + 1) * self.qvel_dim]

            base_pos = base_pos_all[n]

            if self.cc_cfg.action_v == 1:# base_pos (69, ) for one agent
                while np.any(base_pos - qpos[7:] > np.pi):
                    base_pos[base_pos - qpos[7:] > np.pi] -= 2 * np.pi
                while np.any(base_pos - qpos[7:] < -np.pi):
                    base_pos[base_pos - qpos[7:] < -np.pi] += 2 * np.pi
            elif self.cc_cfg.action_v == 0:
                base_pos = cfg.a_ref
                raise NotImplementedError

            target_pos = base_pos + ctrl_joint

            k_p = np.zeros(qvel.shape[0])
            k_d = np.zeros(qvel.shape[0])

            if cfg.meta_pd:
                # for one agent this is:
                # self.ndof = 69, self.vf_dim = 216, self.meta_pd_dim = 30, self.sim_iter = 15
                pd_start = (self.ndof + self.vf_dim)
                pd_end = (self.ndof + self.vf_dim + self.meta_pd_dim)
                meta_pds = ctrl[pd_start:pd_end] # goes from 285:315 --> (30, )
                # self.jkp is (141,), is that ok?  i_iter=number of skip frames
                curr_jkp = self.jkp.copy() * np.clip((meta_pds[i_iter] + 1), 0, 10)
                curr_jkd = self.jkd.copy() * np.clip((meta_pds[i_iter + self.sim_iter] + 1), 0, 10)
                # if flags.debug:
                # import ipdb; ipdb.set_trace()
                # print((meta_pds[i_iter + self.sim_iter] + 1), (meta_pds[i_iter] + 1))
            elif cfg.meta_pd_joint:
                num_jts = self.jkp.shape[0]
                meta_pds = ctrl[(self.ndof + self.vf_dim):(self.ndof + self.vf_dim + self.meta_pd_dim)]
                curr_jkp = self.jkp.copy() * np.clip((meta_pds[:num_jts] + 1), 0, 10)
                curr_jkd = self.jkd.copy() * np.clip((meta_pds[num_jts:] + 1), 0, 10)
            else:
                curr_jkp = self.jkp.copy()
                curr_jkd = self.jkd.copy()

            k_p[6:] = curr_jkp
            k_d[6:] = curr_jkd
            qpos_err = np.concatenate((np.zeros(6), qpos[7:] + qvel[6:] * dt - target_pos)) # (75, )
            qvel_err = qvel
            
            qpos_err_all.append(qpos_err)
            qvel_err_all.append(qvel_err)
            k_p_all.append(k_p)
            k_d_all.append(k_d)
            curr_jkp_all.append(curr_jkp)
            curr_jkd_all.append(curr_jkd)

        qpos_err_all = np.concatenate(qpos_err_all, axis=0) # (150, )
        qvel_err_all = np.concatenate(qvel_err_all, axis=0) # (150, )
        k_p_all = np.concatenate(k_p_all, axis=0) # (150, )
        k_d_all = np.concatenate(k_d_all, axis=0) # (150, )
        curr_jkd_all = np.concatenate(curr_jkd_all, axis=0) # (138, )
        curr_jkp_all = np.concatenate(curr_jkp_all, axis=0) # (138, )
        # todo multi: this has to be computed for both agents at the same time
        q_accel = self.compute_desired_accel(qpos_err_all, qvel_err_all, k_p_all, k_d_all) # (150, )
        qvel_err_all += q_accel * dt  # (150, )
        ### treat qvel_err_all[6:]
        qvel_err1_6 = qvel_err_all[:self.qvel_dim][6:]
        qvel_err2_6 = qvel_err_all[self.qvel_dim:][6:]
        qvel_err_6 = np.concatenate((qvel_err1_6, qvel_err2_6), axis=0)
        #####
        # this was another mistake, not doing this for qpos and repeating qvel_err_6 to compute
        # the torque
        # big mistake!
        # qpos_err1_6 = qpos_err_all[:self.qpos_dim][6:] # (70, ) --> should be 69
        # qpos_err2_6 = qpos_err_all[self.qpos_dim:][6:] # (68, ) --> should be 69
        # 75 because qpos_err is 75 w/ the first 6 elements being 0
        qpos_err1_6 = qpos_err_all[:75][6:] # (69, ) --> should be 69
        qpos_err2_6 = qpos_err_all[75:][6:] # (69, ) --> should be 69
        qpos_err_6 = np.concatenate((qpos_err1_6, qpos_err2_6), axis=0) # (138, )
        # this was also wrong, mistaking curr_jkp_all for curr_jkd_all in the first term
        torque = -curr_jkp_all * qpos_err_6 - curr_jkd_all * qvel_err_6 # (138, )
        # print(curr_jkp_all[69:])
        # print(qpos_err_6[69:])
        # print(curr_jkd_all[69:])
        # print(qvel_err_6[69:])
        # print(torque[69:])
        return torque

    """ RFC-Explicit """

    def rfc_explicit(self, vf, vf_bodies, vf_geoms, qfrc, agent_id=None):
        """for Multi agent, possibly can be done per agent, but have to pass vf_bodies"""
        # vf_geoms is different per agent
        # for agent 1, vf_geoms = [1, 2, 6, 10, 3, 7, 11, 4, 8, 12, 5, 9, 13, 15, 20, 14, 16, 21, 17, 22, 18, 23, 19, 24]
        # for agent 2, vf_geoms = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48]
        # also vf_bodies
        assert agent_id is not None, "agent_id must be specified"

        n = agent_id
        # qfrc = np.zeros(self.data.qfrc_applied.shape) # (150, )
        all_forces = []
        all_forces_w = []
        all_torques = []
        all_torques_w = []
        all_body_xmat = []
        all_body_xpos = []
        num_each_body = self.cc_cfg.get("residual_force_bodies_num", 1) # int = 1
        residual_contact_only = self.cc_cfg.get("residual_contact_only", False) # boolean = False
        residual_contact_only_ground = self.cc_cfg.get("residual_contact_only_ground", False) # boolean = False
        residual_contact_projection = self.cc_cfg.get("residual_contact_projection", False) # boolean = False
        vf_return = np.zeros(vf.shape) # (216, )
        # agent 1 vf_bodies:
        # agent 2 vf_bodies: ('Pelvis_2', 'L_Hip_2', 'L_Knee_2', 'L_Ankle_2', 'L_Toe_2', 'R_Hip_2', 'R_Knee_2', 'R_Ankle_2', 'R_Toe_2', 'Torso_2', 'Spine_2', 'Chest_2', 'Neck_2', 'Head_2', 'L_Thorax_2', 'L_Shoulder_2', 'L_Elbow_2',

        # xpos1 = self.data.body_xpos[-24:]
        # xpos2 = self.data.body_xpos[1:-24]
        # print(xpos1)
        # print(xpos2)
        for i, body in enumerate(vf_bodies): # body is the name of the body, e.g., "Pelvis"
            body_id = self.model._body_name2id[body] # body_id 0-49: 0=world, 1-24=agent1, 25-49=agent2
            foot_pos = self.data.get_body_xpos(body)[2] # returns the z coord of the body xpos
            has_contact = False
            # checks this geom collision with all the contact geoms, possibly can be done per agent
            # self.vf_geoms here is wrong, should change per agent
            geom_id = self.vf_geoms[i]
            # contact documentation: https://mujoco.readthedocs.io/en/2.2.2/computation.html#collision
            # why is self.data.ncon = 13
            # contact documentation: https://mujoco.readthedocs.io/en/2.2.2/programming.html#contacts
            for contact in self.data.contact[:self.data.ncon]: # self.data.contact is (500, ), self.data.ncon=13?
                g1, g2 = contact.geom1, contact.geom2
                if (g1 == 0 and g2 == geom_id) or (g2 == 0 and g1 == geom_id):
                    has_contact = True
                    break

            if residual_contact_only_ground:
                pass
            else:
                has_contact = foot_pos <= 0.12

            if not (residual_contact_only and not has_contact):
                for idx in range(num_each_body):
                    contact_point = vf[(i * num_each_body + idx) * self.body_vf_dim:(i * num_each_body + idx) * self.body_vf_dim + 3]
                    if residual_contact_projection:
                        contact_point = self.smpl_robot[n].project_to_body(body, contact_point)
                    # force and torque are (3,)
                    force = (vf[(i * num_each_body + idx) * self.body_vf_dim + 3:(i * num_each_body + idx) * self.body_vf_dim + 6] * self.cc_cfg.residual_force_scale)
                    torque = (vf[(i * num_each_body + idx) * self.body_vf_dim + 6:(i * num_each_body + idx) * self.body_vf_dim + 9] * self.cc_cfg.residual_force_scale if self.cc_cfg.residual_force_torque else np.zeros(3))
                    all_forces.append(force)
                    all_torques.append(torque)
                    # contact_point is (3,)
                    contact_point = self.pos_body2world(body, contact_point)
                    force_w = self.vec_body2world(body, force)
                    torque_w = self.vec_body2world(body, torque)
                    # self.data.get_body_xmat(body)
                    # id = self.model.body_name2id(body)

                    # body_xmat is (3, 3), they are Frame orientations
                    # From docu: The results of forward kinematics are availabe in mjData as xpos, xquat and xmat for bodies
                    # it is weird that xmat is different for normal and swap as qpos is the exact same for both
                    # From docu: https://mujoco.readthedocs.io/en/stable/programming/simulation.html?highlight=xmat#coordinate-frames-and-transformations
                    # The quantities in mjData that start with “x” are expressed in global coordinates.
                    # These are mjData.xpos, mjData.geom_xpos etc. Frame orientations are usually stored as 3-by-3
                    # matrices (xmat), except for bodies whose orientation is also stored as a unit quaternion mjData.xquat
                    body_xmat = self.data.get_body_xmat(body)
                    all_body_xmat.append(body_xmat)
                    body_xpos = self.data.get_body_xpos(body)
                    all_body_xpos.append(body_xpos)

                    all_forces_w.append(force_w)
                    all_torques_w.append(torque_w)
                    # vf_return gets filled up here at each pass
                    vf_return[(i * num_each_body + idx) * self.body_vf_dim:(i * num_each_body + idx) * self.body_vf_dim + 3] = contact_point
                    vf_return[(i * num_each_body + idx) * self.body_vf_dim + 3:(i * num_each_body + idx) * self.body_vf_dim + 6] = (force_w / self.cc_cfg.residual_force_scale)

                    # print(np.linalg.norm(force), np.linalg.norm(torque))
                    # todo multi: can this be a source of error? it is done here per agent, should it be done for all at once?
                    mjf.mj_applyFT(
                        self.model, # const mjModel* m
                        self.data, # mjData* d, adds the result to the vector mjData.qfrc_applied?
                        force_w, # const mjtNum force[3],
                        torque_w, # const mjtNum torque[3],
                        contact_point, #  const mjtNum point[3],
                        body_id, # int body,
                        qfrc, # mjtNum* qfrc_target,  (150, ) this gets filled each by by 3 places. 3 elements are set each time
                    )
        # what is self.curr_vf? and why vf_return is assigned to it?
        # this should be taken outside
        # self.curr_vf = vf_return  # (216, ) this size is per agent!
        # self.data.qfrc_applied[:] = qfrc

        # all_forces = np.concatenate(all_forces, axis=0)
        # all_torques = np.concatenate(all_torques, axis=0)
        # all_forces_w = np.concatenate(all_forces_w, axis=0)
        # all_torques_w = np.concatenate(all_torques_w, axis=0)
        # all_body_xmat = np.concatenate(all_body_xmat, axis=0)
        # all_body_xpos = np.concatenate(all_body_xpos, axis=0)

        # print(all_body_xpos)
        # print(self.data.qpos[:76])
        # print(self.data.qpos[76:])
        if 0:
            from utils.smpl import from_qpos_to_verts_save
            from utils.misc import save_pointcloud
            inspect_path = "inspect_out/rfc_explicit/"
            lim = 76
            xpos = all_body_xpos.reshape(-1, 3)

            qpos = self.data.qpos[lim:]
            from_qpos_to_verts_save(qpos, self, inspect_path, out_fname="qpos_normal.ply", agent_id=1)
            save_pointcloud(xpos, inspect_path + f"xpos_normal.ply")

            qpos = self.data.qpos[:lim]
            from_qpos_to_verts_save(qpos, self, inspect_path, out_fname="qpos_swapped.ply", agent_id=0)
            save_pointcloud(xpos, inspect_path + f"xpos_swapped.ply")


        # print(all_forces)
        # print(all_torques)
        # print(all_forces_w)
        # print(all_torques_w)
        # print(all_body_xmat)
        # print(all_body_xpos)

        return qfrc, vf_return

    def do_simulation(self, action, n_frames):
        t0 = time.time()
        cfg = self.cc_cfg
        ctrl_all = action  # for one agent (315, ), this comes from cc_a_all --> UHC output
        # import ipdb; ipdb.set_trace()

        # meta_pds = ctrl[(self.ndof + self.vf_dim):(self.ndof + self.vf_dim +
        #                                            self.meta_pd_dim)]
        # print(np.max(meta_pds), np.min(meta_pds))

        #####################  simulation for loop for n_frames #########################
        self.curr_torque = []
        for i in range(n_frames):
            if cfg.action_type == "position":
                torque = self.compute_torque(ctrl_all, i_iter=i)  # for two agents: (138, )
            elif cfg.action_type == "torque":
                torque = ctrl_all * self.a_scale * 100
            # modified self.torque_lim for two agents
            torque = np.clip(torque, -self.torque_lim, self.torque_lim)
            # all_body_xpos0 = get_xpos(0, self.model.body_names, self.data)
            # all_body_xpos1 = get_xpos(1, self.model.body_names, self.data)
            # print(all_body_xpos0)
            # print(all_body_xpos1)
            # torque[(self.get_expert_kin_pose() == 0)] = 0
            self.curr_torque.append(torque)
            # NUK: very hacky. change
            # self.data.ctrl[:] = torque
            self.data.ctrl[:138] = torque  # data.ctrl is (69, ) for one agent and (138, ) for two agents
            # print(torque[:69])
            # print(torque[69:])
            # print(self.data.qpos[76:])
            # ctrl_ = np.concatenate(ctrl)
            """ Residual Force Control (RFC) """
            qfrc_all = []
            vf_return_all = []
            qfrc = np.zeros(self.data.qfrc_applied.shape)  # (150, )

            for n in range(self.num_agents):
                if cfg.residual_force:
                    # it ENTERS HERE, vf is part of cc_a_all for each agent separate
                    vf = ctrl_all[n][(self.ndof):(self.ndof + self.vf_dim)].copy()  # vf is (216, )
                    if cfg.residual_force_mode == "implicit":
                        raise NotImplementedError
                        self.rfc_implicit(vf)
                    else:
                        # in default eval_scene it comes here
                        # does this has to be done for all agents?
                        # e.g., bodies from agent 2: 2_Chest

                        # Stupid mistake!
                        # vf_bodies = self.vf_bodies if n==0 else self.model.body_names[-24:]
                        vf_bodies = self.vf_bodies if n==0 else [c+"_2" for c in self.vf_bodies]
                        body_id_list = self.model.geom_bodyid.tolist()
                        vf_geoms = [body_id_list.index(self.model._body_name2id[body]) for body in vf_bodies]
                        # for agent 1, vf_geoms = [1, 2, 6, 10, 3, 7, 11, 4, 8, 12, 5, 9, 13, 15, 20, 14, 16, 21, 17, 22, 18, 23, 19, 24]
                        # for agent 2, vf_geoms = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48]
                        # multi: vf_geoms[i] is changed here to be per agent
                        qfrc, vf_return = self.rfc_explicit(vf, vf_bodies, vf_geoms, qfrc, agent_id=n)
                        qfrc_all.append(qfrc)
                        vf_return_all.append(vf_return)

            # todo multi: check dims here, do not look right
            # qfrc_all_ is (300, ) --> this has to be (150,)
            # so take only the first half from [0] and second half from [1]

            # qfrc_all_ = np.zeros_like(qfrc_all[0])
            # qfrc_all_[:self.qvel_dim] = qfrc_all[0][:self.qvel_dim]
            # qfrc_all_[self.qvel_dim:] = qfrc_all[1][self.qvel_dim:]
            # print(qfrc)
            # print(qfrc[:75])
            # print(qfrc[75:])
            # print(qfrc_all[0][:75])
            # print(qfrc_all[1][75:])
            # print(vf)
            # print(vf_return_all[1])
            # print(vf_return_all[0])
            vf_return_all = np.concatenate(vf_return_all, axis=0)  # (432, )
            # self.curr_vf = vf_return
            # self.data.qfrc_applied[:] = qfrc
            #################### This is important! #########################################
            self.curr_vf = vf_return_all
            # self.data.qfrc_applied[:] = qfrc_all_ # qfrc_applied --> should be (150, ) for two agents
            # TODO: activar de nuevo el RFC
            self.data.qfrc_applied[:] = qfrc # should be (150, )

                        # if flags.debug:
            #     self.data.qpos[: self.qpos_lim] = self.get_expert_qpos(
            #         delta_t=-1
            #     )  # debug
            #     self.sim.forward()  # debug

            self.sim.step()

            if 0:
                for t in range(1, 5):
                    from pathlib import Path
                    img = self.sim.render(width=400, height=400, camera_name=f"camera{i}")
                    # plot(img)
                    # img = rot_img(img, show=False)
                    path = Path(f"inspect_out/sim_render/chi3d/humanoid_im/sim_cam{t}/%03d.png" % i)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    save_img(path, img)

        ######################################################################################

        if self.viewer is not None:
            self.viewer.sim_time = time.time() - t0


    def load_models(self):
        self.converter = SMPLConverter(
            self.smpl_model,
            self.sim_model,
            smpl_model=self.cc_cfg.robot_cfg.get("model", "smpl"),
        )
        # all these have already two agent information, when using multi
        self.sim_iter = 15
        self.qpos_lim = self.converter.get_new_qpos_lim() # 152
        self.qvel_lim = self.converter.get_new_qvel_lim() # 150
        self.body_lim = self.converter.get_new_body_lim() # 49
        self.jpos_diffw = self.converter.get_new_diff_weight()[:, None] # 24
        self.body_diffw = self.converter.get_new_diff_weight()[1:] # 23
        self.body_qposaddr = get_body_qposaddr(self.model) # 48
        # what is self.jkd?
        # todo multi: adapt to multi but overriding he method in MultiHumanoidEnv
        self.jkd = self.converter.get_new_jkd() * self.cc_cfg.get("pd_mul", 1) # for one agent (69, )
        self.jkp = self.converter.get_new_jkp() * self.cc_cfg.get("pd_mul", 1) # for one agent (69, )

        self.a_scale = self.converter.get_new_a_scale() # for one agent (69, )
        torque_lim = self.converter.get_new_torque_limit() * self.cc_cfg.get("tq_mul", 1) # for one agent (69, )
        torque_lim_multi = np.concatenate((torque_lim, torque_lim), axis=0) # for two agents (138, )
        self.torque_lim = torque_lim_multi
        self.set_action_spaces()



if __name__ == "__main__":
    pass
