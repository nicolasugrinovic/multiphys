'''
Modified from EmbodiedPose repo: https://github.com/ZhengyiLuo/EmbodiedPose
This is the env used for the eval_scene, it seems that here is where everything happens: the simulation and the forward step

'''


import os
import os.path as osp
import sys
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

from uhc.utils.torch_geometry_transforms import (angle_axis_to_rotation_matrix as aa2mat,
                                                 rotation_matrix_to_angle_axis as mat2aa)
import json
import copy

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
    # set the colors of each object
    colors = np.empty(voxelarray.shape, dtype=object)
    colors[voxelarray] = 'red'
    # and plot everything
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(voxelarray, facecolors=colors, edgecolor='k')
    plt.show()


class MultiHumanoidKinEnvRes(HumanoidEnv):
    """THIS HANDLE MULTIPLE ROBOTS AND INHERIT FROM HumanoidEnv BUT HAVE
     MODIFIED METHODS
     This class is responsible for MPG, so it uses the 3D projected kpts and compares w/ the 2D inputs
     The object of this class should be created once per scene NOT per agent

     The agents are defined here. The body shapes are determined here. Look at: reset_robot()
     """

    # Wrapper class that wraps around Copycat agent from UHC

    def __init__(self, kin_cfg, init_context, cc_iter=-1, mode="train", agent=None):

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
        self.simulate = True
        self.voxel_thresh = 0.1
        self.next_frame_idx = 250
        self.op_thresh = 0.1
        self.load_context_pass = 0
        self.pred_joints2d = [[] for i in range(self.num_agents)]
        self.smpl_robot = []
        self.use_quat = cc_cfg.robot_cfg.get("ball", False)
        cc_cfg.robot_cfg['span'] = kin_cfg.model_specs.get("voxel_span", 1.8)

        self.smpl_robot_orig = SceneRobotMulti(cc_cfg.robot_cfg, data_dir=osp.join(cc_cfg.base_dir, "data/smpl"))
        # This contains one HB object per agent
        self.hb = [Humanoid_Batch(data_dir=osp.join(cc_cfg.base_dir, "data/smpl")) for _ in range(self.num_agents)]

        ############################# Agent loaded here #########################################
        # This is the list of robots, one per agent
        self.smpl_robot = [SceneRobotMulti(
            cc_cfg.robot_cfg,
            data_dir=osp.join(cc_cfg.base_dir, "data/smpl"),
            masterfoot=cc_cfg.masterfoot,
            num_agent=1
        ) for _ in range(self.num_agents)]
        ##########################################################################################
        if self.num_agents > 1:
            # better pass the list of robots
            self.merge_agents_xml(self.smpl_robot)

        ## This XML already contains the 2 agents ##
        self.xml_str = self.smpl_robot[0].export_xml_string().decode("utf-8")  # here xml create w/out betas info
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
            self.j2d_seq_feat = [collections.deque([0] * self.num_context, self.num_context) for i in
                                 range(self.num_agents)]

        self.body_grad = np.zeros(63)
        self.bm = bm = self.motion_prior.bm_dict['neutral']
        self.smpl2op_map = smpl_to_openpose(bm.model_type, use_hands=False, use_face=False, use_face_contour=False,
                                            openpose_format='coco25')  # this is used by the Humanoid_Batch class I think
        self.smpl_2op_submap = self.smpl2op_map[self.smpl2op_map < 22]


        # the mujoco env is initialized here with the xml_str
        mujoco_env.MujocoEnv.__init__(self, self.xml_str, frame_skip=15)

        self.prev_qpos = self.data.qpos.copy()  # (76, )--> this is now (152,)
        self.setup_constants(cc_cfg, cc_cfg.data_specs, mode=mode, no_root=False)
        self.neutral_path = self.kin_cfg.data_specs['neutral_path']
        self.neutral_data = joblib.load(self.neutral_path)
        ###############################
        self.load_context(init_context)  # here BETAS info should be added to the xml file
        ###############################
        self.set_action_spaces()
        self.set_obs_spaces()
        self.weight = mujoco_py.functions.mj_getTotalmass(self.model)

        ''' Load CC Controller '''
        cc_obs = self.get_cc_obs()[0]
        self.state_dim = state_dim = cc_obs.shape[0]
        cc_action_dim = self.action_dim  # originally 315, should be 315*2=630? -->maybe not as this sets
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

    def setup_constants(self, cfg, data_specs, mode, no_root):
        """ Overrides function from HumanoidEnv"""
        # NUK: this function was added here by me, it overrides method from HumanoidEnv
        self.cc_cfg = cfg
        self.set_cam_first = set()
        # todo. maybe i'll have to modify smpl_robot_orig to contain 2 agents
        self.smpl_model = load_model_from_xml(self.smpl_robot_orig.export_xml_string().decode("utf-8"))

        self.sim_model = load_model_from_xml(
            self.smpl_robot[0].export_xml_string().decode("utf-8"))  # state of the loaded simulation
        self.expert = None
        self.base_rot = data_specs.get("base_rot", [0.7071, 0.7071, 0.0, 0.0])
        self.netural_path = data_specs.get("neutral_path", "sample_data/standing_neutral.pkl")
        self.no_root = no_root
        self.body_diff_thresh = cfg.get("body_diff_thresh", 0.5)
        self.body_diff_thresh_test = cfg.get("body_diff_thresh_test", 0.5)
        self.mode = mode
        self.end_reward = 0.0
        self.start_ind = 0
        self.rfc_rate = 1 if not cfg.rfc_decay else 0
        self.prev_bquat = None
        self.load_models()
        self.set_model_base_params()
        self.bquat = self.get_body_quat()
        # this should vary with the number of agents
        self.humanoid = [Humanoid(model=self.model) for _ in range(self.num_agents)]
        self.curr_vf = None  # storing current vf
        self.curr_torque = None  # Strong current applied torque at each joint

    def explcude_contacts(self, tree1_contacts, tree_agent2_copy):
        contact_node = tree1_contacts
        agent1_body_nodes_list = tree_agent1_copy.getroot().find("worldbody").findall(".//body")
        agent2_body_nodes_list = tree_agent2_copy.getroot().find("worldbody").findall(".//body")
        assert len(agent1_body_nodes_list) == len(
            agent2_body_nodes_list) == 24, "Expecting 24 body nodes when excluding contacts"
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

    def merge_agents_xml(self, smpl_robot_list):
        """ add a second agent to the xml tree.
        here tree2 is from agent2 and self.tree should be from agent1

        """
        tree = smpl_robot_list[0].tree
        tree_agent1_copy = copy.deepcopy(smpl_robot_list[0].tree)
        # generate copy of the tree but with modified names according to agent id
        remove_elements = ["equality"]
        new_trees = []
        # get the xml nodes for each agent
        for nn in range(len(smpl_robot_list) - 1):
            tree2, _ = smpl_robot_list[nn].export_robot_str(remove_elements=remove_elements, agent_id=nn + 2)
            new_trees.append(tree2)

        # copy nodes so that we don't overwrite them when manipulating the tree
        new_agent_trees = []
        for nn in range(len(smpl_robot_list) - 1):
            tree_agent2 = copy.deepcopy(new_trees[nn])
            new_agent_trees.append(tree_agent2)

        worldbody = tree.getroot().find("worldbody")

        # process NODES - the assets from the additional agent will be placed under the asset node
        tree1_asset = tree.getroot().find("asset")  # save original file assets
        # get assets per agent
        for nn in range(len(smpl_robot_list) - 1):
            agent2_asset = new_trees[nn].getroot().find("asset")
            for asset in agent2_asset:
                tree1_asset.append(asset)

        # append BODY per new agent
        for nn in range(len(smpl_robot_list) - 1):
            agent2_body_node = new_trees[nn].getroot().find("worldbody").find("body")
            worldbody.append(agent2_body_node)

        ########## append contact node to the worldbody node
        # append CONTACT per new agent
        tree1_contacts = tree.getroot().find("contact")
        for nn in range(len(smpl_robot_list) - 1):
            # agent2_contacts = tree_agent2.getroot().find("contact")
            agent2_contacts = new_trees[nn].getroot().find("contact")
            for contact in agent2_contacts:
                tree1_contacts.append(contact)
                # print(asset.tag)

        ########################################################################################################
        # exclude pelvis between agents
        if self.kin_cfg.exclude_contacts:
            self.explcude_contacts(tree1_contacts, tree_agent2_copy)
        ########################################################################################################

        # append ACTUATOR per new agent
        tree1_actuators = tree.getroot().find("actuator")
        for nn in range(len(smpl_robot_list) - 1):
            agent2_actuator = new_trees[nn].getroot().find("actuator")
            for actuator in agent2_actuator:
                tree1_actuators.append(actuator)

        print(f'* Merged xml files! for {len(smpl_robot_list)} agents')

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


            if "obj_info" in context:
                obj_info = context['obj_info']
                self.smpl_robot[0].load_from_skeleton(torch.from_numpy(beta[0:1, :]).float(), gender=gender,
                                                      obj_info=obj_info)
            else:
                # this seems important for loading the scene
                if not context.get("load_scene", True):
                    scene_name = None
                ####################################################################################################
                # This generates the MESHES used in the simulation
                self.smpl_robot[n].load_from_skeleton(torch.from_numpy(beta[0:1, :]).float(), gender=gender,
                                                      scene_and_key=scene_name, num_agent=1)

                ####################################################################################################
        self.merge_agents_xml(self.smpl_robot)
        xml_str = self.smpl_robot[0].export_xml_string().decode("utf-8")  # get the xml_str for onlly one agent

        ######################################
        self.reload_sim_model(xml_str)
        ######################################
        self.weight = self.smpl_robot[0].weight
        self.proj_2d_loss = []
        self.proj_2d_body_loss = []
        self.proj_2d_root_loss = []
        self.proj_2d_line_loss = []

        for n in range(self.num_agents):
            beta = self.context_dict[n]["beta"].copy()
            self.hb[n].update_model(torch.from_numpy(beta[0:1, :16]), torch.tensor(gender[0:1]))
            self.hb[n].update_projection(self.camera_params, self.smpl2op_map, MUJOCO_2_SMPL)
            # Losses for the MGP (multi_step_grad) are initialized here!
            self.proj_2d_loss.append(egrad(self.hb[n].proj_2d_loss))
            self.proj_2d_body_loss.append(egrad(self.hb[n].proj_2d_body_loss))
            self.proj_2d_root_loss.append(egrad(self.hb[n].proj_2d_root_loss))
            self.proj_2d_line_loss.append(egrad(self.hb[n].proj_2d_line_loss))
        return xml_str  # this return is ignored

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
        self.camera_params_torch = {k: torch.from_numpy(v).double() if isNpArray(v) else v for k, v in
                                    self.camera_params.items()}

        ######################################
        self.reset_robot()
        self.target = []
        self.gt_targets = []
        self.prev_humor_state = []
        self.cur_humor_state = []
        self.pred_tcn = []

        # iterare PER agent data
        for n, context in enumerate(self.context_dict):
            ######################################
            self.humanoid[n].update_model(self.model, n_agent=n)
            ######################################

            context['len'] = context['pose_aa'].shape[0] - 1
            # todo multi: use smpl_to_qpose_multi!
            gt_qpos = smpl_to_qpose_multi(context['pose_aa'], self.model, trans=context['trans'],
                                          count_offset=True, agent_id=n)

            # this is the HUMOR estimated pose
            init_qpos = smpl_to_qpose_multi(context['init_pose_aa'][None,], self.model,
                                            trans=context['init_trans'][None,],
                                            count_offset=True, agent_id=n)
            context["qpos"] = gt_qpos

            # multi: this target should have 2 people
            target = self.humanoid[n].qpos_fk(torch.from_numpy(init_qpos))

            prev_humor_state = {k: data_dict[n][k][:, 0:1, :].clone() for k in self.motion_prior.data_names}
            cur_humor_state = prev_humor_state
            #####
            gt_targets = self.humanoid[n].qpos_fk(torch.from_numpy(gt_qpos))
            # Initializing target
            target.update({k: data_dict[n][k][:, 0:1, :].clone() for k in self.motion_prior.data_names})

            if self.kin_cfg.model_specs.get("use_tcn", False):
                # this is the HUMOR pose to convert it to world coordinates
                world_body_pos = target['wbpos'].reshape(24, 3)[MUJOCO_2_SMPL][self.smpl_2op_submap]

                world_trans = world_body_pos[..., 7:8:, :]
                pred_tcn = {
                    'world_body_pos': world_body_pos - world_trans,
                    'world_trans': world_trans,
                }

                casual = self.kin_cfg.model_specs.get("casual_tcn", True)
                full_R, full_t = self.camera_params["full_R"], self.camera_params['full_t']

                if casual:  # in the demo specs casual_tcn: true
                    joints2d = context["joints2d"][0:1].copy()  # shape (1, 12, 3)
                    joints2d[joints2d[..., 2] < self.op_thresh] = 0  # op_thresh=0.1
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
        self.init_qpos = self.sim.data.qpos.copy()
        self.init_qvel = self.sim.data.qvel.copy()
        self.viewer = None
        self._viewers = {}


    def set_model_params(self):
        if self.cc_cfg.action_type == 'torque' and hasattr(self.cc_cfg, 'j_stiff'):
            self.model.jnt_stiffness[1:] = self.cc_cfg.j_stiff
            self.model.dof_damping[6:] = self.cc_cfg.j_damp

    def get_obs(self):
        ar_obs = self.get_ar_obs_v1()
        return ar_obs

    def get_cc_obs(self):
        assert self.cc_cfg.obs_type == 'full' and self.cc_cfg.obs_v == 2, \
            "ERROR: Should choose another get_full_obs_v from HumanoidEnv - IMPLEMENT!"
        return self.get_full_obs_v2()

    def get_full_obs_v2(self, delta_t=0):
        data = self.data
        self.qpos_dim = 76  # is this correct?
        self.qvel_dim = 75
        self.xpos_dim = 24
        qpos_all = data.qpos[:self.qpos_lim].copy()  # (152, )
        qvel_all = data.qvel[:self.qvel_lim].copy()  # (150, )

        target_body_qpos_all = self.get_expert_qpos(
            delta_t=1 + delta_t)  # nos (76,) is that ok?  # target body pose (1, 76)
        target_quat_all = self.get_expert_wbquat(delta_t=1 + delta_t)  # .reshape(-1, 4) # (96,)
        target_jpos_all = self.get_expert_joint_pos(delta_t=1 + delta_t)  # (72,)
        body_xpos_all = self.data.body_xpos.copy()[1:]  # remove world body xpos, left with (48, 3)
        cur_quat_all = self.data.body_xquat.copy()[1:]  # remove world body, (48, 4)

        obs_all = []
        for n in range(self.num_agents):
            qpos_start = self.qpos_dim * n
            qpos_end = self.qpos_dim * (n + 1)
            qvel_start = self.qvel_dim * n
            qvel_end = self.qvel_dim * (n + 1)
            xpos_start = self.xpos_dim * n
            xpos_end = self.xpos_dim * (n + 1)

            qpos = qpos_all[qpos_start:qpos_end].copy()  # should be ndarray (76?, )
            qvel = qvel_all[qvel_start:qvel_end].copy()  # should be ndarray (75?, )

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
            target_body_qpos = target_body_qpos_all[n]  # target body pose (1, 76)# it is ok if it's (76,)
            target_quat = target_quat_all[n].reshape(-1, 4)  # (24, 4)
            target_jpos = target_jpos_all[n]  # (72,)
            ################ Body pose and z ################
            target_root_quat = self.remove_base_rot(target_body_qpos[3:7])  # (4,)

            qpos[3:7] = de_heading(curr_root_quat)  # deheading the root, (76,)
            diff_qpos = target_body_qpos.copy()
            diff_qpos[2] -= qpos[2]  # compute the difference in z
            diff_qpos[7:] -= qpos[7:]  # compute the difference in joint rotations
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
            obs.append(np.array([rel_h]))  # (1,) obs: heading difference in root angles

            # ZL: this is wrong. Makes no sense. Should be target_root_pos. Should be fixed.
            rel_pos = target_root_quat[:3] - qpos[:3]
            rel_pos = transform_vec(rel_pos, curr_root_quat, self.cc_cfg.obs_coord).ravel()
            obs.append(rel_pos[:2])  # (2,) obs: relative x, y difference (1, 2)

            ################ target/difference joint positions ################
            curr_jpos = body_xpos_all[xpos_start:xpos_end]  # this is now (24, 3)

            # translate to body frame (zero-out root)
            r_jpos = curr_jpos - qpos[None, :3]
            r_jpos = transform_vec_batch(r_jpos, curr_root_quat, self.cc_cfg.obs_coord)  # body frame position
            # obs: target body frame joint position (1, 72)
            obs.append(r_jpos.ravel())  # (72,) obs: target body frame joint position
            diff_jpos = target_jpos.reshape(-1, 3) - curr_jpos
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
                obs.append(shape_gender_obs)  # (17,) shape_gender_obs

            obs = np.concatenate(obs)  # is it ok for obs to be (657, )
            obs_all.append(obs)
        return obs_all

    def get_ar_obs_v1(self):
        t = self.cur_t

        curr_qpos_all = self.data.qpos[:self.qpos_lim].copy()
        curr_qvel = self.data.qvel[:self.qvel_lim].copy()
        self.prev_humor_state = copy.deepcopy(self.cur_humor_state)
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
            curr_root_quat = self.remove_base_rot(curr_qpos[3:7])  # (4,)
            full_R, full_t = self.camera_params_torch['full_R'], self.camera_params_torch['full_t']
            target_global_dict = {
                k: torch.from_numpy(self.context_dict[n][k][(t + 1):(t + 2)].reshape(humor_dict[n][k].shape))
                for k in self.motion_prior.data_names}
            conon_output = self.motion_prior.canonicalize_input_double(humor_dict[n], target_global_dict,
                                                                       split_input=False, return_info=True)
            humor_local_dict, next_target_local_dict, info_dict = conon_output
            # print(torch.matmul(humor_dict['trans'], full_R.T) + full_t)
            # info_dict --> keys (['world2aligned_trans', 'world2aligned_rot', 'trans2joint'])
            heading_rot = info_dict['world2aligned_rot'].numpy()  # (1, 3, 3)
            curr_body_obs = np.concatenate(
                [humor_local_dict[k].flatten().numpy() for k in self.motion_prior.data_names])
            # curr_body_obs # (336,)
            # hq = get_heading_new(curr_qpos[3:7])
            hq = 0
            obs.append(np.array([hq]))  # (1,)
            obs.append(curr_body_obs)  # (336,)
            if compute_root_obs:
                is_root_obs.append(np.array([1]))
                is_root_obs.append(
                    np.concatenate([[1 if "root" in k else 0] * humor_local_dict[k].flatten().numpy().shape[-1]
                                    for k in self.motion_prior.data_names]))

            if self.kin_cfg.model_specs.get("use_tcn", False):
                casual = self.kin_cfg.model_specs.get("casual_tcn", True)
                if casual:  # it enters here
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
                        joints2d_gt = np.pad(joints2d_gt, ([0, pad_num - joints2d_gt.shape[0]], [0, 0], [0, 0]),
                                             mode="edge")

                    joints2d_gt[..., :2] = normalize_screen_coordinates(joints2d_gt[..., :2],
                                                                        self.camera_params['img_w'],
                                                                        self.camera_params['img_h'])
                    joints2d_gt[joints2d_gt[..., 2] < self.op_thresh] = 0  # (12, 3)


                if self.kin_cfg.model_specs.get("tcn_3dpos", False):
                    cam_pred_3d = humor_dict[n]['cam_pred_3d']
                    cam_pred_3d = smpl_op_to_op(cam_pred_3d)
                    if casual:
                        j2d3dfeat = np.concatenate([joints2d_gt[..., :2], cam_pred_3d.squeeze()], axis=1)
                        self.j2d_seq_feat[n].append(j2d3dfeat)  # push next step obs into state
                    else:
                        j2d3dfeat = np.concatenate([joints2d_gt[..., :2],
                                                    np.repeat(cam_pred_3d.squeeze(1), self.num_context // 2 + 1,
                                                              axis=0)], axis=-1)
                        [self.j2d_seq_feat[n].pop() for _ in range(self.num_context // 2)]
                        [self.j2d_seq_feat[n].append(feat) for feat in j2d3dfeat]
                ########################## NUK: it enters here ############################################
                else:
                    if casual:  # what is j2d_seq_feat?
                        self.j2d_seq_feat[n].append(joints2d_gt[:, :2])  # (12, 2)  # push next step obs into state
                    else:
                        [self.j2d_seq_feat[n].pop() for _ in range(self.num_context // 2)]
                        [self.j2d_seq_feat[n].append(feat) for feat in joints2d_gt[..., :2]]
                ###########################################################################################
                j2d_seq = np.array(self.j2d_seq_feat[n]).flatten()  # np.array of (1944,)
                obs.append(j2d_seq)  # j2d_seq shape: (1944,) are 81 flattened 12 joints2d, 12*2*81 = 1944
                if compute_root_obs:
                    vari = np.array([3] * j2d_seq.shape[0])
                    is_root_obs.append(vari)

                # use tcn directly on the projection gradient
                tcn_root_grad = self.kin_cfg.model_specs.get("tcn_root_grad", False)  # boolean
                world_body_pos, world_trans = self.pred_tcn[n]['world_body_pos'], self.pred_tcn[n][
                    'world_trans']  # (14, 3) and (1, 3)
                curr_body_jts = humor_dict[n]['joints'].reshape(22, 3)[self.smpl_2op_submap].numpy()  # (14, 3)
                curr_body_jts -= curr_body_jts[..., 7:8, :]  # root relative?
                world_body_pos -= world_body_pos[..., 7:8, :]  # ndarray (14, 3)
                body_diff = transform_vec_batch_new(world_body_pos - curr_body_jts,
                                                    curr_root_quat).T.flatten()  # ndarray (42, )

                if self.kin_cfg.model_specs.get("tcn_body", False):
                    obs.append(body_diff)
                # todo: target shoudl also be for 2 agents
                curr_trans = self.target[n]['wbpos'][:, :3]  # ndarray (1, 3) # this is in world coord
                trans_diff = np.matmul(world_trans - curr_trans, heading_rot[0].T).flatten()  # ndarray (3,)
                trans_diff[2] = world_trans[:, 2]  # Mimicking the target trans feat.
                if self.kin_cfg.model_specs.get("tcn_traj", False):
                    obs.append(trans_diff)

                if not tcn_root_grad:  # enters
                    pred_root_mat = op_to_root_orient(world_body_pos[None,])  # ndarray (1, 3, 3)
                    root_rot_diff = np.matmul(heading_rot, pred_root_mat).flatten()  # ndarray (9, )
                    obs.append(root_rot_diff)  # (9, )

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
                root_grad = (sRot.from_matrix(heading_rot) * sRot.from_rotvec(
                    proj2dgrad[3:6] / body_mul)).as_rotvec().squeeze()
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
                no_grad_body = self.kin_cfg.model_specs.get("no_grad_body", False)  # boolean
                proj2dgrad = humor_dict[n]['proj2dgrad'].squeeze().numpy().copy()  # ndarray (75, )
                proj2dgrad = np.nan_to_num(proj2dgrad, nan=0, posinf=0, neginf=0)
                proj2dgrad = np.clip(proj2dgrad, -200, 200)
                # sRot if from scipy.spatial.transform rotation
                trans_grad = (np.matmul(heading_rot, proj2dgrad[:3])).squeeze()  # ndarray (3, )
                root_grad = (sRot.from_matrix(heading_rot) * sRot.from_rotvec(
                    proj2dgrad[3:6])).as_rotvec().squeeze()  # ndarray (3, )
                body_grad = proj2dgrad[6:69]  # ndarray (63, )
                if no_grad_body:
                    # Ablation, zero body grad. Just TCN
                    body_grad = np.zeros_like(body_grad)
                obs.append(trans_grad)  # (3,)
                if compute_root_obs:
                    is_root_obs.append(np.array([1] * trans_grad.shape[0]))
                obs.append(root_grad)  # (3,)
                if compute_root_obs:
                    is_root_obs.append(np.array([1] * root_grad.shape[0]))
                obs.append(body_grad)  # (63,)
                if compute_root_obs:
                    is_root_obs.append(np.array([1] * body_grad.shape[0]))
            ################################################################################################################

            if self.kin_cfg.model_specs.get("use_sdf", False):
                sdf_vals = self.smpl_robot[n].get_sdf_np(self.cur_humor_state[n]['joints'].reshape(-1, 3), topk=3)
                obs.append(sdf_vals.numpy().flatten())
                if compute_root_obs:
                    is_root_obs.append(np.array([2] * sdf_vals.shape[0]))
            elif self.kin_cfg.model_specs.get("use_dir_sdf", False):
                sdf_vals, sdf_dirs = self.smpl_robot[n].get_sdf_np(self.cur_humor_state[n]['joints'].reshape(-1, 3),
                                                                   topk=3, return_grad=True)
                sdf_dirs = np.matmul(sdf_dirs, heading_rot[0].T)  # needs to be local dir coord
                sdf_feat = (sdf_vals[:, :, None] * sdf_dirs).numpy().flatten()
                obs.append(sdf_feat)
                if compute_root_obs:
                    is_root_obs.append(np.array([2] * sdf_feat.shape[0]))
            ################ VOXEL observations ############################################################################
            ########################### it enters here #####################################################################
            if self.kin_cfg.model_specs.get("use_voxel", False):
                voxel_res = self.kin_cfg.model_specs.get("voxel_res", 8)  # this is =16
                # these voxel_feat are float continuous values
                voxel_feat = self.smpl_robot[n].query_voxel(self.cur_humor_state[n]['trans'].reshape(-1, 3),
                                                            self.cur_humor_state[n]['root_orient'].reshape(3, 3),
                                                            res=voxel_res).flatten()  # (4096,)
                # these are booleans of shape (4096,) and self.voxel_thresh=0.1
                inside, outside = voxel_feat <= 0, voxel_feat >= self.voxel_thresh


                middle = np.logical_and(~inside, ~outside)
                # voxel_feat has values different from 0 and 1 due to middle
                voxel_feat[inside], voxel_feat[outside] = 1, 0
                voxel_feat[middle] = (self.voxel_thresh - voxel_feat[middle]) / self.voxel_thresh
                if compute_root_obs:
                    is_root_obs.append(np.array([2] * voxel_feat.shape[0]))
                obs.append(voxel_feat)  # (4096,)
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

            obs = np.concatenate(obs)  # (6455,)
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
        self.target = [[] for i in range(self.num_agents)]  # overwrites previous target values

        for n in range(self.num_agents):
            action = action_all[n].copy()
            action_ = torch.from_numpy(action[None, :69])
            next_global_out = self.motion_prior.step_state(self.cur_humor_state[n], action_)
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
                gt_pose_ = gt_pose.reshape(-1, 21, 3, 3).reshape(-1, 21 * 3 * 3)
                next_global_out['pose_body'] = gt_pose_[self.cur_t].reshape(1, 1, -1)

            # multi: this should be multi-agent
            qpos = smpl_to_qpose_torch_multi(pose_aa, self.model, trans=next_global_out['trans'].reshape(1, 3),
                                             count_offset=True, agent_id=n)  # (1, 76)

            # HUMANOID is used here, HUMANOID needs to be multi-agent
            # multi: make use of humanoid multi-agent
            if self.mode == "train" and self.agent.iter < self.agent.num_supervised and self.agent.iter >= 0:
                # Dagger
                qpos = torch.from_numpy(self.gt_targets[n]['qpos'][(self.cur_t):(self.cur_t + 1)])
                fk_res = self.humanoid[n].qpos_fk(qpos)
            else:
                fk_res = self.humanoid[n].qpos_fk(qpos)  # dict of 15 elements

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
                world_body_pos = cam_body_pos.dot(full_R)
                self.pred_tcn[n]['world_body_pos'] = world_body_pos
                self.pred_tcn[n]['cam_body_pos'] = cam_trans + cam_body_pos


    def get_humanoid_pose_aa_trans(self, qpos=None, agent_id=None):
        assert agent_id is not None, "agent_id must be specified"
        if qpos is None:
            qpos = self.data.qpos.copy()[None]
        pose_aa, trans = qpos_to_smpl_multi(qpos, self.model, self.cc_cfg.robot_cfg.get("model", "smpl"), agent_id)
        # qpos is (2, 76), pose_aa is (2, 24, 3), trans is (2, 3)
        return pose_aa, trans

    def get_humor_dict_obs_from_sim(self):
        """ NUK: Compute obs based on current and previous simulation state and coverts it into humor format. """
        qpos_dim_one = 76
        num_agents = self.num_agents
        qpos_all = self.data.qpos.copy()[None]  # (1, 76) for 1 agent, (1, 152) for 2 agents, (1, 228) for 3 agents
        prev_qpos_all = self.prev_qpos[None]  # (1, 76)

        all_humor = []
        for n in range(num_agents):
            start = n * qpos_dim_one
            end = (n + 1) * qpos_dim_one
            qpos = qpos_all[:, start: end]  # (1, 76)
            prev_qpos = prev_qpos_all[:, start: end]  # (1, 76)
            # # NUK hack for now
            # qpos = qpos[:, :76]
            # prev_qpos = prev_qpos[:, :76]
            # Calculating the velocity difference from simulation. We do not use target velocity.
            qpos_stack = np.concatenate([prev_qpos, qpos])
            # No need to be multi inside get_humanoid_pose_aa_trans
            pose_aa, trans = self.get_humanoid_pose_aa_trans(qpos_stack, agent_id=n)  # Simulation state.

            fk_result = self.humanoid[n].qpos_fk(torch.from_numpy(qpos_stack), to_numpy=False, full_return=False)
            trans_batch = torch.from_numpy(trans[None])  # ([1, 2, 3])

            joints = fk_result["wbpos"].reshape(-1, 24, 3)[:, MUJOCO_2_SMPL].reshape(-1, 72)[:, :66]  # (2,, 66)
            pose_aa_mat = aa2mat(torch.from_numpy(pose_aa.reshape(-1, 3))).reshape(1, 2, 24, 4, 4)[..., :3,
                          :3]  # ([1, 2, 24, 3, 3])
            trans_vel, joints_vel, root_orient_vel = estimate_velocities(trans_batch, pose_aa_mat[:, :, 0],
                                                                         joints[None],
                                                                         30, aa_to_mat=False)

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
            joints2d_gt = self.context_dict[n]['joints2d'][t:(t + grad_frame_num)].copy()  # (1, 12, 3)

            if joints2d_gt.shape[0] < grad_frame_num:
                try:
                    joints2d_gt = np.pad(joints2d_gt, ([0, grad_frame_num - joints2d_gt.shape[0]], [0, 0], [0, 0]),
                                         mode="edge")
                except:
                    print('bug!')
            inliers = joints2d_gt[..., 2] > self.op_thresh  # boolean: (1, 12)
            self.hb[n].update_tgt_joints(joints2d_gt[..., :2], inliers)

            # input_vect contains the SMPL pose corresponding to current qpos only
            input_vec = np.concatenate([humor_out['trans'].numpy(), pose_aa[1:2].reshape(1, -1, 72)],
                                       axis=2)  # (1, 1, 75)

            ######################################## Projection of 3D to 2D keypoints ######################################
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
                pose_grad, input_vec_new, curr_loss = self.multi_step_grad(input_vec, order=order,
                                                                           num_adpt_grad=num_adpt_grad,
                                                                           normalize=normalize, step_size=grad_step,
                                                                           agent_id=n)
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
                pose_grad, input_vec_new, curr_loss = self.multi_step_grad(input_vec,
                                                                           order=order,
                                                                           num_adpt_grad=num_adpt_grad,
                                                                           normalize=normalize,
                                                                           step_size=grad_step,
                                                                           agent_id=n
                                                                           )
                ############################################################################################################
                multi = depth / 10
                pose_grad[:6] *= multi
                humor_out["proj2dgrad"] = pose_grad  # (1, 1, 75)

            all_humor.append(humor_out)

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
        if tcn_root_grad:  # does not enter here
            world_body_pos, world_trans = self.pred_tcn['world_body_pos'], self.pred_tcn['world_trans']
            pred_root_vec = sRot.from_matrix(op_to_root_orient(world_body_pos[None,])).as_rotvec()  # tcn's root
            input_vec_new[..., 3:6] = pred_root_vec

        if order == 1:
            step_size = 0.00001
            step_size_a = step_size * np.clip(prev_loss, 0, 5)
        else:
            if normalize:  # does not normalize
                step_size_a = step_size / 1.02
            else:  # it enters here
                step_size_a = 0.000005
        for iteration in range(num_adpt_grad):  # num_adpt_grad=5
            # it enters the if
            if self.kin_cfg.model_specs.get("use_3d_grad_sept", False):  # enters here
                proj2dgrad_body = self.proj_2d_body_loss[agent_id](input_vec_new, ord=order, normalize=normalize)
                proj2dgrad = self.proj_2d_loss[agent_id](input_vec_new, ord=order, normalize=normalize)
                proj2dgrad[..., 3:] = proj2dgrad_body[..., 3:]
                proj2dgrad = np.nan_to_num(proj2dgrad, posinf=0,
                                           neginf=0)  # This is essentail, otherwise nan will get more
            else:
                proj2dgrad = self.proj_2d_loss[agent_id](input_vec_new, ord=order, normalize=normalize)
                proj2dgrad = np.nan_to_num(proj2dgrad, posinf=0,
                                           neginf=0)  # This is essentail, otherwise nan will get more

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
        fail = False
        cfg = self.kin_cfg
        cc_cfg = self.cc_cfg
        self.prev_qpos = self.get_humanoid_qpos()  # now (152, )
        self.prev_qvel = self.get_humanoid_qvel()  # now (150, )
        self.prev_bquat = self.bquat.copy()  # now (188, ) ??
        self.prev_hpos = self.get_head().copy()  # now (7, ) ??
        self.step_ar(a)

        # do N inters without updating the target kinematic pose, a sort of "inner loop"
        for inner_iters in range(self.kin_cfg.loops_uhc):
            ####################################### UHC step ##############################################
            cc_a_all = []
            # calls self.get_full_obs_v2(), for multi-agent it overrides super().get_obs()
            cc_obs_all = self.get_cc_obs()  # runs super().get_obs() # cc_obs is (657, )?? , list of N where N is the number of agents
            # this will loop over each agent, where n is the agent id
            for n, cc_obs in enumerate(cc_obs_all):
                # this runs ZFilter() from uhc/khrylib/utils/zfilter.py --> does y = (x-mean)/std
                cc_obs = self.cc_running_state(cc_obs, update=False)
                ########################################### CC step ##########################################
                cc_a = self.cc_policy.select_action(torch.from_numpy(cc_obs)[None,], mean_action=True)[0].numpy()
                cc_a_all.append(cc_a)
                ##############################################################################################

            ################################ Physical simulation occurs here ##############################################
            if flags.debug:
                self.do_simulation(cc_a_all, self.frame_skip)
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
                            self.do_simulation(cc_a_all, self.frame_skip)
                        except Exception as e:
                            import traceback
                            print(f"Exception in do_simulation, at iter: {self.cur_t},  with error: \n {e}")
                            print(traceback.format_exc())
                            fail = True
                    else:
                        # debug
                        expert_qpos = self.get_expert_qpos()
                        self.data.qpos[:self.qpos_lim] = np.concatenate(expert_qpos)
                        self.sim.forward()  # debug

            pass
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

        print('**** Setting state for N AGENTS ****')
        qpose_comb = np.concatenate(qpos)
        qvel_comb = np.concatenate(qvel)

        assert qpose_comb.shape == (self.model.nq,) and qvel_comb.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(
            old_state.time, qpose_comb, qvel_comb, old_state.act, old_state.udd_state
        )
        self.sim.set_state(new_state)
        self.sim.forward()

    def reset_model(self, qpos=None, qvel=None):
        cfg = self.kin_cfg
        ind = 0
        self.start_ind = 0  # what is this???

        self.qpos_dim = 76  # is this correct?
        self.qvel_dim = 75
        self.xpos_dim = 24

        if qpos is None:
            init_qpos_all = []
            init_vel_all = []
            for n, context_dict in enumerate(self.context_dict):
                init_pose_aa = context_dict['init_pose_aa']
                init_trans = context_dict['init_trans']
                init_qpos = smpl_to_qpose_multi(torch.from_numpy(init_pose_aa[None,]),
                                                self.model, torch.from_numpy(init_trans[None,]),
                                                count_offset=True, agent_id=n).squeeze()
                init_vel = np.zeros(self.qvel_dim)
                init_qpos_all.append(init_qpos)  # init_qpos is (76, ) ?
                init_vel_all.append(init_vel)  # init_qpos is (75, ) ?

        else:
            init_qpos_all = qpos
            init_vel_all = qvel
        #######################
        self.set_state(init_qpos_all, init_vel_all)
        #######################
        self.prev_qpos = self.get_humanoid_qpos()  # 152, and 152/2 = 76
        ################################### GET OBS #################################
        obs = self.get_obs()
        #############################################################################
        return obs

    def viewer_setup(self, mode):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.lookat[:2] = self.get_humanoid_qpos()[:2]

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

    def get_target_kin_pose(self, delta_t=0):  #
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
            wbquat = self.target[n]['wbquat'].squeeze()  # (96,)
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

    def compute_torque(self, ctrl_all, i_iter=0):  # i_iter=number of skip frames
        """ NUK: this is multi agent version of the original compute_torque from HumanoidEnv
        ctrl_all: are the actions that come from the UHC policy
        """
        cfg = self.cc_cfg
        dt = self.model.opt.timestep
        qpos_all = self.get_humanoid_qpos()  # (152, ) for 2 agents, (228, ) for 3 agents
        qvel_all = self.get_humanoid_qvel()  # (150, ) for 2 agents, (225, ) for 3 agents

        # multi: made multi agent version
        if self.cc_cfg.action_v == 1:
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
        curr_jkD_all = []
        for n in range(self.num_agents):
            # n=0 # --> mistake
            ctrl = ctrl_all[n]  # (315, ) where does this come from?
            ctrl_joint = ctrl[:self.ndof]
            qpos = qpos_all[n * self.qpos_dim: (n + 1) * self.qpos_dim]
            qvel = qvel_all[n * self.qvel_dim: (n + 1) * self.qvel_dim]
            base_pos = base_pos_all[n]

            if self.cc_cfg.action_v == 1:  # base_pos (69, ) for one agent
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
                meta_pds = ctrl[pd_start:pd_end]  # goes from 285:315 --> (30, )
                # self.jkp is (141,), is that ok?  i_iter=number of skip frames
                curr_jkp = self.jkp.copy() * np.clip((meta_pds[i_iter] + 1), 0, 10)
                curr_jkd = self.jkd.copy() * np.clip((meta_pds[i_iter + self.sim_iter] + 1), 0, 10)
            elif cfg.meta_pd_joint:
                num_jts = self.jkp.shape[0]
                meta_pds = ctrl[(self.ndof + self.vf_dim):(self.ndof + self.vf_dim + self.meta_pd_dim)]
                curr_jkp = self.jkp.copy() * np.clip((meta_pds[:num_jts] + 1), 0, 10)
                curr_jkd = self.jkd.copy() * np.clip((meta_pds[num_jts:] + 1), 0, 10)
            else:
                curr_jkp = self.jkp.copy()
                curr_jkd = self.jkd.copy()

            k_p[6:] = curr_jkp  # curr_jkp  should be (69, ) for any N of agents
            k_d[6:] = curr_jkd  # curr_jkd should be (69, ) for any N of agents
            qpos_err = np.concatenate((np.zeros(6), qpos[7:] + qvel[6:] * dt - target_pos))  # (75, )
            qvel_err = qvel

            qpos_err_all.append(qpos_err)
            qvel_err_all.append(qvel_err)
            k_p_all.append(k_p)
            k_d_all.append(k_d)
            curr_jkp_all.append(curr_jkp)
            curr_jkD_all.append(curr_jkd)

        qpos_err_all = np.concatenate(qpos_err_all, axis=0)  # (150, )
        qvel_err_all = np.concatenate(qvel_err_all, axis=0)  # (150, )
        k_p_all = np.concatenate(k_p_all, axis=0)  # (150, )
        k_d_all = np.concatenate(k_d_all, axis=0)  # (150, )
        curr_jkD_all = np.concatenate(curr_jkD_all, axis=0)  # (138, )
        curr_jkp_all = np.concatenate(curr_jkp_all, axis=0)  # (138, )
        # todo multi: this has to be computed for both agents at the same time
        q_accel = self.compute_desired_accel(qpos_err_all, qvel_err_all, k_p_all,
                                             k_d_all)  # (150, ) for 2 agents, (225, ) for 3 agents
        qvel_err_all += q_accel * dt  # (150, ) for 2 agents, (225, ) for 3 agents
        qvel_err_6_all = []
        for nn in range(self.num_agents):
            start = nn * self.qvel_dim
            end = (nn + 1) * self.qvel_dim
            qvel_err_6 = qvel_err_all[start:end][6:]
            qvel_err_6_all.append(qvel_err_6)
        qvel_err_6 = np.concatenate(qvel_err_6_all, axis=0)
        #####
        qpos_err_6_all = []
        for nn in range(self.num_agents):
            start = nn * self.qvel_dim
            end = (nn + 1) * self.qvel_dim
            qpos_err_6 = qpos_err_all[start:end][6:]
            qpos_err_6_all.append(qpos_err_6)

        qpos_err_6 = np.concatenate(qpos_err_6_all, axis=0)  # (138, )
        torque = -curr_jkp_all * qpos_err_6 - curr_jkD_all * qvel_err_6  # (138, )
        return torque

    """ RFC-Explicit """

    def rfc_explicit(self, vf, vf_bodies, vf_geoms, qfrc, agent_id=None):
        """for Multi agent, possibly can be done per agent, but have to pass vf_bodies"""
        assert agent_id is not None, "agent_id must be specified"
        n = agent_id
        # qfrc = np.zeros(self.data.qfrc_applied.shape) # (150, )
        all_forces = []
        all_forces_w = []
        all_torques = []
        all_torques_w = []
        all_body_xmat = []
        all_body_xpos = []
        num_each_body = self.cc_cfg.get("residual_force_bodies_num", 1)  # int = 1
        residual_contact_only = self.cc_cfg.get("residual_contact_only", False)  # boolean = False
        residual_contact_only_ground = self.cc_cfg.get("residual_contact_only_ground", False)  # boolean = False
        residual_contact_projection = self.cc_cfg.get("residual_contact_projection", False)  # boolean = False
        vf_return = np.zeros(vf.shape)  # (216, )
        for i, body in enumerate(vf_bodies):  # body is the name of the body, e.g., "Pelvis"
            body_id = self.model._body_name2id[
                body]  # body_id 0-49: 0=world, 1-24=agent1, 25-49=agent2, 50-73=agent3 but this is done per agent so maybe we dont care
            foot_pos = self.data.get_body_xpos(body)[2]  # returns the z coord of the body xpos
            has_contact = False
            geom_id = self.vf_geoms[i]
            for contact in self.data.contact[:self.data.ncon]:  # self.data.contact is (500, ), self.data.ncon=13?
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
                    contact_point = vf[(i * num_each_body + idx) * self.body_vf_dim:(
                                                                                                i * num_each_body + idx) * self.body_vf_dim + 3]
                    if residual_contact_projection:
                        contact_point = self.smpl_robot[n].project_to_body(body, contact_point)
                    # force and torque are (3,)
                    force = (vf[(i * num_each_body + idx) * self.body_vf_dim + 3:(
                                                                                             i * num_each_body + idx) * self.body_vf_dim + 6] * self.cc_cfg.residual_force_scale)
                    torque = (vf[(i * num_each_body + idx) * self.body_vf_dim + 6:(
                                                                                              i * num_each_body + idx) * self.body_vf_dim + 9] * self.cc_cfg.residual_force_scale if self.cc_cfg.residual_force_torque else np.zeros(
                        3))
                    all_forces.append(force)
                    all_torques.append(torque)
                    # contact_point is (3,)
                    contact_point = self.pos_body2world(body, contact_point)
                    force_w = self.vec_body2world(body, force)
                    torque_w = self.vec_body2world(body, torque)
                    body_xmat = self.data.get_body_xmat(body)
                    all_body_xmat.append(body_xmat)
                    body_xpos = self.data.get_body_xpos(body)
                    all_body_xpos.append(body_xpos)

                    all_forces_w.append(force_w)
                    all_torques_w.append(torque_w)
                    # vf_return gets filled up here at each pass
                    vf_return[(i * num_each_body + idx) * self.body_vf_dim:(
                                                                                       i * num_each_body + idx) * self.body_vf_dim + 3] = contact_point
                    vf_return[(i * num_each_body + idx) * self.body_vf_dim + 3:(
                                                                                           i * num_each_body + idx) * self.body_vf_dim + 6] = (
                                force_w / self.cc_cfg.residual_force_scale)

                    # print(np.linalg.norm(force), np.linalg.norm(torque))
                    # todo multi: can this be a source of error? it is done here per agent, should it be done for all at once?
                    mjf.mj_applyFT(
                        self.model,  # const mjModel* m
                        self.data,  # mjData* d, adds the result to the vector mjData.qfrc_applied?
                        force_w,  # const mjtNum force[3],
                        torque_w,  # const mjtNum torque[3],
                        contact_point,  # const mjtNum point[3],
                        body_id,  # int body,
                        qfrc,
                        # mjtNum* qfrc_target = (150, ) this gets filled each by by 3 places. 3 elements are set each time
                    )

        return qfrc, vf_return

    def do_simulation(self, action, n_frames):
        t0 = time.time()
        cfg = self.cc_cfg
        ctrl_all = action  # for one agent (315, ), this comes from cc_a_all --> UHC output

        #####################  simulation for loop for n_frames #########################
        self.curr_torque = []
        for i in range(n_frames):
            if cfg.action_type == "position":
                torque = self.compute_torque(ctrl_all, i_iter=i)  # for two agents: (138, )
            elif cfg.action_type == "torque":
                torque = ctrl_all * self.a_scale * 100
            # modified self.torque_lim for two agents
            torque = np.clip(torque, -self.torque_lim, self.torque_lim)
            self.curr_torque.append(torque)
            self.data.ctrl[
            :69 * self.num_agents] = torque  # data.ctrl is (69, ) for one agent and (138, ) for two agents
            """ Residual Force Control (RFC) """
            qfrc_all = []
            vf_return_all = []
            qfrc = np.zeros(
                self.data.qfrc_applied.shape)  # (N*75, )--> (150, ) for two agents, (225, ) for three agents

            for n in range(self.num_agents):
                if cfg.residual_force:
                    # it ENTERS HERE, vf is part of cc_a_all for each agent separate
                    vf = ctrl_all[n][(self.ndof):(self.ndof + self.vf_dim)].copy()  # vf is (216, )
                    if cfg.residual_force_mode == "implicit":
                        raise NotImplementedError
                        self.rfc_implicit(vf)
                    else:

                        vf_bodies = self.vf_bodies if n == 0 else [c + f"_{n + 1}" for c in
                                                                   self.vf_bodies]  # this seems to not change vf_bodies
                        body_id_list = self.model.geom_bodyid.tolist()
                        vf_geoms = [body_id_list.index(self.model._body_name2id[body]) for body in vf_bodies]
                        qfrc, vf_return = self.rfc_explicit(vf, vf_bodies, vf_geoms, qfrc, agent_id=n)
                        qfrc_all.append(qfrc)
                        vf_return_all.append(vf_return)

            vf_return_all = np.concatenate(vf_return_all, axis=0)  # (432, )
            #################### This is important! #########################################
            self.curr_vf = vf_return_all
            self.data.qfrc_applied[:] = qfrc  # should be (150, )
            self.sim.step()
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
        self.qpos_lim = self.converter.get_new_qpos_lim()  # 152 (for 2), 228 (for 3)
        self.qvel_lim = self.converter.get_new_qvel_lim()  # 150 (for 2), 225 (for 3)
        self.body_lim = self.converter.get_new_body_lim()  # 49 (for 2), 73 (for 3)
        self.jpos_diffw = self.converter.get_new_diff_weight()[:, None]  # 24 (for 2), 48 (for 3)
        self.body_diffw = self.converter.get_new_diff_weight()[1:]  # 23 (for 2), 47 (for 3)
        self.body_qposaddr = get_body_qposaddr(self.model)  # 48 (for 2), 72 (for 3)
        self.jkd = self.converter.get_new_jkd() * self.cc_cfg.get("pd_mul",
                                                                  1)  # (69, ) --> should be 69 for any N of agents
        self.jkp = self.converter.get_new_jkp() * self.cc_cfg.get("pd_mul",
                                                                  1)  # (69, ) --> should be 69 for any N of agents

        self.a_scale = self.converter.get_new_a_scale()  # for one agent (69, ) | (141,) for 3
        torque_lim = self.converter.get_new_torque_limit() * self.cc_cfg.get("tq_mul", 1)  # (69, )
        torque_lim_multi = np.concatenate([torque_lim for i in range(self.num_agents)], axis=0)  # (138, )
        self.torque_lim = torque_lim_multi
        self.set_action_spaces()


if __name__ == "__main__":
    pass
