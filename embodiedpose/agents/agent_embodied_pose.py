'''
File: /agent_scene.py
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

from unittest import loader
import joblib
import os.path as osp
import pdb
import sys
import glob
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm
import pickle
from collections import defaultdict
import multiprocessing
import math
import time
import os
import torch
import wandb
import gc
import time

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.append(os.getcwd())

from uhc.khrylib.models.mlp import MLP
from uhc.khrylib.rl.core import estimate_advantages
from uhc.khrylib.utils.torch import *
from uhc.khrylib.utils.memory import Memory
from uhc.khrylib.rl.core.critic import Value
from uhc.utils.flags import flags
from uhc.envs import env_dict
from uhc.agents.agent_uhm import AgentUHM
from uhc.smpllib.smpl_eval import compute_metrics
from uhc.utils.math_utils import smpl_op_to_op
from uhc.utils.tools import CustomUnpickler

from embodiedpose.data_loaders import data_dict
from embodiedpose.envs import env_dict
from embodiedpose.models import policy_dict
from embodiedpose.core.reward_function import reward_func
from embodiedpose.core.trajbatch_humor import TrajBatchHumor
from embodiedpose.models.humor.utils.humor_mujoco import MUJOCO_2_SMPL
from embodiedpose.models.humor.utils.humor_mujoco import OP_14_to_OP_12

from PIL import Image
from utils.misc import plot
from utils.misc import save_img
# from pyquaternion import Quaternion as Q
from pathlib import Path

import yaml

# R_mat = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
import datetime
# ct stores current time
ct = datetime.datetime.now()
data_time = ct.strftime("%Y-%m-%d_%H-%M-%S")

def rot_img(img, show=False):
    im = Image.fromarray(img)
    angle = 180
    out = im.rotate(angle)
    out_rot = np.asarray(out)
    # out.save('rotate-output.png')
    if show:
        plot(out_rot)
    return out_rot

class AgentEmbodiedPose(AgentUHM):

    def __init__(self, cfg, dtype, device, mode="train", checkpoint_epoch=0):
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.mode = mode

        self.global_start_fr = 0
        self.iter = checkpoint_epoch
        self.num_warmup = 300
        self.num_supervised = 5
        self.fr_num = self.cfg.fr_num

        ###################################################
        self.setup_vars()
        self.setup_data_loader()
        self.setup_policy()
        ###################################################
        # the agent is loaded here and self.env is created
        self.setup_env()
        # self.env
        ###################################################
        self.setup_value()
        self.setup_optimizer()
        self.setup_logging()
        self.setup_reward()
        self.seed(cfg.seed)
        self.print_config()

        if checkpoint_epoch > 0:
            self.load_checkpoint(checkpoint_epoch)
        elif checkpoint_epoch == -1:
            self.load_curr()

        self.freq_dict = defaultdict(list)
        self.fit_single = False
        self.load_scene = self.cfg.model_specs.get('load_scene', False)
        # already no body
        if 0:
            fname = "inspect_out/xml_s/robot2_AgentEmbodiedPose_init_middle.xml"
            self.env.smpl_robot2.write_xml(fname)

        # from uhc.agents.agent_uhm import AgentUHM
        super(AgentUHM, self).__init__(
            env=self.env,
            dtype=dtype,
            device=device,
            # running_state=ZFilter((self.env.state_dim, ), clip=5),
            running_state=None,
            custom_reward=self.expert_reward,
            mean_action=cfg.render and not cfg.show_noise,
            render=cfg.render,
            num_threads=cfg.num_threads,
            data_loader=self.data_loader,
            policy_net=self.policy_net,
            value_net=self.value_net,
            optimizer_policy=self.optimizer_policy,
            optimizer_value=self.optimizer_value,
            opt_num_epochs=cfg.policy_specs['num_optim_epoch'],
            gamma=cfg.policy_specs['gamma'],
            tau=cfg.policy_specs['tau'],
            clip_epsilon=cfg.policy_specs['clip_epsilon'],
            policy_grad_clip=[(self.policy_net.parameters(), 40)],
            end_reward=cfg.policy_specs['end_reward'],
            use_mini_batch=False,
            mini_batch_size=0)

        # import ipdb; ipdb.set_trace()
        # if self.iter == 0 and self.mode == "train":
        # self.train_init()
        # self.train_init()

    def setup_vars(self):
        super().setup_vars()

        self.traj_batch = TrajBatchHumor

    def setup_reward(self):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        self.expert_reward = expert_reward = reward_func[cfg.policy_specs['reward_id']]

    def setup_env(self):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        """load CC model"""
        # get data
        with torch.no_grad():
            # dataloader is embodiedpose.data_loaders.scene_pose_dataset.ScenePoseDataset
            # dict if 17 fields
            data_sample = self.data_loader.sample_seq(fr_num=20, fr_start=self.global_start_fr)
            if len(data_sample) == 2:
                data_sample = data_sample[0]
            # dict if 23 fields
            # I think policy_net will be used one per agent, so this is ok
            context_sample = self.policy_net.init_context(data_sample, random_cam=True)
        # save config in self
        self.cc_cfg = cfg.cc_cfg
        # chose_env = env_dict[self.cfg.env_name]

        ######################################## Init Env ######################################
        # env_dict = {"kin_res": HumanoidKinEnvRes}
        # here self.smpl_robot is created and mujoco_env.MujocoEnv.__init__(self, self.xml_str, 15) is called
        # todo. create here the new multi-robot env, it should replace the current one
        # options: kin_res, multi_kin_res
        self.env = env_dict[self.cfg.env_name](cfg, init_context=context_sample,
                                               cc_iter=cfg.policy_specs.get('cc_iter', -1),
                                               mode="train", agent=self)

        # here the agent is already defined and in a T-pose lying in the floor
        if 0:
            import mujoco_py
            from mujoco_py import MjViewer
            model = self.env.model
            sim = mujoco_py.MjSim(model)
            viewer = MjViewer(sim)
            viewer.render()
        ##########################################################################################
        self.env.seed(cfg.seed)

    def setup_policy(self):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        data_sample = self.data_loader.sample_seq(fr_num=20, fr_start=self.global_start_fr)
        # todo. maybe modify later
        if len(data_sample)==2:
            data_sample = data_sample[0]
        self.policy_net = policy_net = policy_dict[cfg.policy_name](cfg, data_sample, device=device,
                                                                    dtype=dtype, mode=self.mode,
                                                                    agent=self)
        to_device(device, self.policy_net)

    def setup_value(self):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        state_dim = self.policy_net.state_dim
        action_dim = self.env.action_space.shape[0]
        self.value_net = Value(MLP(state_dim, self.cc_cfg.value_hsize, self.cc_cfg.value_htype))
        to_device(device, self.value_net)

    def setup_data_loader(self):
        cfg = self.cfg
        train_files_path = cfg.data_specs.get("train_files_path", [])
        test_files_path = cfg.data_specs.get("test_files_path", [])
        self.train_data_loaders, self.test_data_loaders, self.data_loader = [], [], None

        if len(train_files_path) > 0:
            for train_file, dataset_name in train_files_path:
                # there are different types of datasets: 'scene_pose', 'video_pose', 'amass', 'amass_multi'
                data_loader = data_dict[dataset_name](cfg, [train_file], multiproess=False)
                self.train_data_loaders.append(data_loader)

        if len(test_files_path) > 0:
            for test_file, dataset_name in test_files_path:
                # dataset is ScenePoseDataset(DatasetBatch)
                data_loader = data_dict[dataset_name](cfg, [test_file], multiproess=False)
                self.test_data_loaders.append(data_loader)

        self.data_loader = np.random.choice(self.train_data_loaders)

    def eval_seq(self, take_key, loader):
        curr_env = self.env

        with to_cpu(*self.sample_modules):
            with torch.no_grad():

                res = defaultdict(list)

                self.policy_net.set_mode('test')
                curr_env.set_mode('test')

                ################
                # this fuction is from scene_pose_dataset.py, a lot happens here!!
                # gets data from the dataset / dataloader
                # context_sample --> keys are:
                # (['joints2d', 'pose_6d', 'pose_aa', 'trans', 'root_orient', 'joints',
                # 'trans_vel', 'root_orient_vel', 'joints_vel', 'pose_body', 'points3d',
                # 'phase', 'time', 'beta', 'gender', 'seq_name', 'cam'])
                # todo I also need to change the data loader to return the j2d for both people
                context_sample = loader.get_sample_from_key(take_key=take_key, full_sample=True, return_batch=True)

                if 0:
                    #inspect the 2D kpts
                    from utils.misc import plot_joints_cv2
                    joints2d = context_sample["joints2d"]
                    joints2d_np = joints2d[0, 0].cpu().numpy()
                    black = np.zeros((900, 900, 3), dtype=np.uint8)
                    plot_joints_cv2(black, joints2d_np[None], with_text=True, return_img=False, sc=2)

                # augments the data with HUMOR initial estimation
                # This function uses "joints" field from context_sample to init the pose
                ########################################################################################################
                # ar_context(): function from the KinPolicyHumorRes class (inherits from KinPolicy) in
                # kin_policy_humor_res.py,
                # HUMOR: this function inits the states of the kin_net, seems to be a wrapper around HUMOR
                # ar_context contains context_sample data augmented with initial 3D pose
                # (most likely init est w/ Humor) and pose data in cam space.
                # it does not process nor changes the data
                # ar_context --> keys are:
                # (['joints2d', 'pose_6d', 'pose_aa', 'trans', 'root_orient', 'joints',
                # 'trans_vel', 'root_orient_vel', 'joints_vel', 'pose_body', 'points3d',
                # 'phase', 'time', 'beta', 'gender', 'seq_name', 'cam',
                # with the additional keys:
                # 'load_scene', 'wbpos_cam', 'root_cam_rot', 'init_pose_aa', 'init_trans', 'scene_name'])
                # todo I also need to change the data loader to return the j2d for both people
                ar_context = self.policy_net.init_context(context_sample, random_cam=(not "cam" in context_sample))
                # points3d = ar_context["points3d"].cpu().numpy()
                # points3d.min()
                # points3d.max()
                # at this point, the agent has not changed pose yet
                # curr_env.render()

                ########################################################################################################
                # initializes the simulation env
                # load_context() is from the HumanoidKinEnvRes class (inherits from HumanoidEnv)
                # in humanoid_kin_res.py (envs)
                # this also add the scene (SceneRobot class)
                # this RESETS robot self.reset_robot(), so each time load_context() is called, the robot is reset
                # todo this needs to be handled once per agent, so the multi-agent env should load context for each sub_env
                # 2nd load_context
                curr_env.load_context(ar_context)
                # at this point, ALSO the agent has not changed pose yet
                # curr_env.render()
                if 0:
                    from utils.smpl import from_qpos_to_smpl
                    from utils.misc import save_trimesh

                    def from_qpos_to_verts_save(gt_qpos, curr_env, inspect_path, out_fname="verts.ply"):
                        # gt_qpos has to be: shape (76,)
                        gt_verts, faces = from_qpos_to_smpl(gt_qpos, curr_env)
                        save_trimesh(gt_verts[0, 0], faces, inspect_path + out_fname)
                    i=1
                    t = 0
                    img = curr_env.sim.render(width=400, height=400, camera_name=f"camera{i}")
                    # plot(img)
                    path = Path(
                        f"inspect_out/sim_render/{self.cfg.data_name}/{self.cfg.seq_name}/{data_time}/sim_cam{i}/%07d.png" % t)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    save_img(path, img)

                    inspect_path = "inspect_out/chi3d/meshes/simu_init/"
                    gt_qpos = curr_env.context_dict['qpos'][curr_env.cur_t] # shape: (76,)
                    target_qpos = curr_env.target['qpos'] # shape: (1, 76)
                    pred_qpos = curr_env.get_humanoid_qpos() # shape: (76,)

                    from_qpos_to_verts_save(target_qpos[0], curr_env, inspect_path, out_fname="target_qpos.ply")
                    from_qpos_to_verts_save(pred_qpos, curr_env, inspect_path, out_fname="pred_qpos.ply")
                    from_qpos_to_verts_save(gt_qpos, curr_env, inspect_path, out_fname="gt_qpos.ply")


                ########################################################################################################
                # the simu is updated here!
                # this state var goes into the kin_policy to predict the residuals
                # for the estimated kin pose
                # here, MPG is called, deep down the rabit , mainly in the obs() function
                # it goes reset_model() --> get_obs() --> get_ar_obs_v1 --> get_humor_dict_obs_from_sim()
                # returns: ob -- meaning it contains the 2d/3d features from the MPG
                # state --> shape (6455,)
                # *** actually the "function get_ar_obs_v1()" builds and returns the ob ***
                # calls uhc/khrylib/rl/envs/common/mujoco_env.py --> MujocoEnv.reset()
                state = curr_env.reset()
                # at this point, the agent HAS CHANGED pose!!
                ########################################################################################################

                # curr_env.render()
                # curr_env.viewer
                # self.env.viewer
                # viewer = curr_env._get_viewer("human")
                # viewer = curr_env._get_viewer("rgb_array")
                # viewer._read_pixels_as_in_window() # RuntimeError: Failed to initialize OpenGL
                # curr_env.model.camera_names
                # curr_env.sim.data.cam_xpos

                if 0:
                    # this is good to visu the simulation at the same time it runs, as opposed to how it
                    # is done in the viewer, however the camera pose is not the same
                    img = curr_env.sim.render(width=400, height=400)
                    img = rot_img(img, show=False)

                if self.running_state is not None:
                    state = self.running_state(state)
                fail = False

                ########################### Starts the FOR LOOP ########################################################
                # one t-step in the simulation env?
                for t in range(10000):

                    if self.cfg.visualizer=="sim_render":
                        for i in range(1, 5):
                            img = curr_env.sim.render(width=400, height=400, camera_name=f"camera{i}")
                            # plot(img)
                            path = Path(f"inspect_out/sim_render/{self.cfg.data_name}/{self.cfg.seq_name}/{data_time}/sim_cam{i}/%07d.png" % t)
                            path.parent.mkdir(parents=True, exist_ok=True)
                            save_img(path, img)

                    # the qpos space is 76D, it is used to represent the pose of the humanoid,
                    # and it can be converted to SMPL pose via utils.smpl.from_qpos_to_smpl()
                    gt_qpos = curr_env.context_dict['qpos'][curr_env.cur_t] # shape: (76,)
                    res['gt'].append(gt_qpos)
                    target_qpos = curr_env.target['qpos'] # shape: (1, 76)
                    res['target'].append(target_qpos)
                    # at time step 0, this is the initial state.
                    # At timestep 1, this is the predicted state at timestep 0 (which corresponds to 0)
                    ########################## predicted pose #######################
                    # this is where the model gets the prediction from the sim env
                    pred_qpos = curr_env.get_humanoid_qpos() # shape: (76,)
                    res['pred'].append(pred_qpos)
                    #################################################################

                    if 0:
                        from utils.smpl import smpl_to_verts
                        from utils.smpl import from_qpos_to_smpl
                        from utils.misc import save_trimesh
                        # def qpos_to_smpl(pred_qpos):
                        #     pred_smpl = curr_env.get_humanoid_pose_aa_trans(pred_qpos[None])
                        #     pred_pose = pred_smpl[0].reshape([1, 72])
                        #     pred_verts, faces = smpl_to_verts(pred_pose, pred_smpl[1])
                        #     return pred_verts, faces
                        inspect_path = f"inspect_out/sim_loop/chi3d/{data_time}/{t:03d}/"
                        pred_verts, faces = from_qpos_to_smpl(pred_qpos, curr_env)
                        save_trimesh(pred_verts[0, 0], faces, inspect_path + f"pred_verts_{t:03d}.ply")
                        gt_verts, faces = from_qpos_to_smpl(gt_qpos, curr_env)
                        save_trimesh(gt_verts[0, 0], faces, inspect_path + f"gt_verts_{t:03d}.ply")
                        target_verts, faces = from_qpos_to_smpl(target_qpos[0], curr_env)
                        save_trimesh(target_verts[0, 0], faces, inspect_path + f"target_verts_{t:03d}.ply")


                    if 'joints_gt' in curr_env.context_dict:
                        res["gt_jpos"].append(smpl_op_to_op(self.env.context_dict['joints_gt'][self.env.cur_t].copy()))
                        res["pred_jpos"].append(smpl_op_to_op(curr_env.get_wbody_pos().copy().reshape(24, 3)[MUJOCO_2_SMPL][curr_env.smpl_2op_submap]))
                    else:
                        # these are 3djoints, though their shape is (72,), they are the 24 joints in 3D space
                        # that correspond to the smpl joints
                        gt_jpos = self.env.gt_targets['wbpos'][self.env.cur_t].copy() # shape: (72,)
                        res["gt_jpos"].append(gt_jpos)
                        pred_jpos = self.env.get_wbody_pos().copy() # shape: (72,)
                        res["pred_jpos"].append(pred_jpos)

                        if 0:
                            from utils.smpl import smpl_to_verts
                            from utils.misc import save_trimesh
                            from utils.misc import save_pointcloud
                            inspect_path = f"inspect_out/sim_loop/chi3d/{data_time}/{t:03d}/"
                            pred_jpos_ = pred_jpos.reshape([24, 3])
                            save_pointcloud(pred_jpos_, inspect_path + f"pred_jpos_{t:03d}.ply")
                            gt_jpos_ = gt_jpos.reshape([24, 3])
                            save_pointcloud(gt_jpos_, inspect_path + f"gt_jpos_{t:03d}.ply")

                    # res["gt_jpos"].append(curr_env.gt_targets['wbpos'][curr_env.cur_t].copy())
                    # res["pred_jpos"].append(curr_env.get_wbody_pos().copy())

                    if self.cfg.model_specs.get("use_tcn", False):
                        # these are 14 joints in 3D with no translation
                        world_body_pos = self.env.pred_tcn['world_body_pos'].copy()[None,] # shape (1, 14, 3)
                        res['world_body_pos'].append(world_body_pos)
                        # at least for the first timestep, this translation is very accurate
                        world_trans = self.env.pred_tcn['world_trans'].copy()[None,] # shape (1, 1, 3)
                        res['world_trans'].append(world_trans)

                        if 0:
                            from utils.misc import save_pointcloud
                            inspect_path = f"inspect_out/sim_loop/chi3d/{data_time}/{t:03d}/"
                            save_pointcloud(world_body_pos[0], inspect_path + f"world_body_pos_{t:03d}.ply")
                            wbody_pos_trans = world_body_pos + world_trans
                            save_pointcloud(wbody_pos_trans[0], inspect_path + f"wbody_pos_trans_{t:03d}.ply")

                    # t_s = time.time()
                    # what is state?
                    state_var = tensor(state).unsqueeze(0).double()
                    # meant to transform the states before inputting to the policy, currently does nothing
                    trans_out = self.trans_policy(state_var)
                    ############################ Kinematic policy #####################################
                    # this policy comes from KinPolicyHumorRes (kin_policy_humor_res.py)
                    # trans_out --> shape ([1, 6455])
                    # action --> shape (114,)
                    # NOTE: the kinematic policy should compute the residual 3D movement, i.e, the state velocity
                    # composed of q_t = (r^R, r^T, theta_t) or root velo in translation, root velo in ori,
                    # velo in joint angles. so what is action? action's first 69 elements are the residuals
                    # that we care about: trans, ori, and pose velos this are applyied to the pose
                    # in the next function --> curr_env.step()
                    # todo. action has to contain actions for both agents. maybe do one pass for each agent
                    action = self.policy_net.select_action(trans_out, mean_action=True)[0].numpy()
                    action = int(action) if self.policy_net.type == 'discrete' else action.astype(np.float64)

                    ############################ The state gets updated here ###########################
                    # **NOTE: MPG is called inside this function!!!!!!!!!!!
                    # this env is HumanoidKinEnvRes from embodiedpose.envs.humanoid_kin_res,
                    # this inherits from UHC - HumanoidEnv(copycat)
                    # next_state --> is obs from curr_env.step(), (6455,)
                    # action --> shape (114,)
                    next_state, env_reward, done, info = curr_env.step(action, kin_override=False)
                    # curr_env will cotain the pred_joints2d
                    # next_state, env_reward, done, info = curr_env.step(action, kin_override=True)
                    ####################################################################################

                    # NUK: added by me
                    if t % 20 == 0:
                        if self.cfg.debug:
                            done = True
                        # print("Step:", t, "Reward:", env_reward, "Action:", action)
                        print(f"Step {t}: is done? {done}, Failed? {info['fail']}, percent {info['percent']}")
                        # just for debugging

                    # # NUK: added by me
                    # if t % 400 == 0 and t!=0:
                    #     done = True

                    if self.cfg.render:
                        curr_env.render()
                    if self.running_state is not None:
                        next_state = self.running_state(next_state)

                    if info['fail']:
                        print("Fail!", take_key)
                        fail = info['fail']

                    # if info['end']: # Always carry till the end
                    if done:
                        out_path = Path(f"inspect_out/sim_render/{self.cfg.data_name}/{self.cfg.seq_name}/{data_time}")
                        out_path.mkdir(parents=True, exist_ok=True)
                        ############################## save videos #####################################
                        if self.cfg.visualizer == "sim_render":
                            for i in range(1, 5):
                                out_path_cams = out_path / Path(f"sim_cam{i}/%07d.png" % t)
                                os.system(
                                    f"ffmpeg -framerate 30 -pattern_type glob -i '{out_path_cams.parent}/*.png' "
                                    f"-c:v libx264 -vf fps=30 -pix_fmt yuv420p {out_path_cams.parent}.mp4 -y")
                                os.system(f"rm -rf {out_path_cams.parent}")

                        article_info = [
                            self.cfg.__dict__,
                        ]

                        with open(f"{out_path}/config.yaml", 'w') as yamlfile:
                            yaml.dump(article_info, yamlfile)
                            # print("Write successful")
                        ###############################################################################

                        ###### When done, collect the last simulated state.
                        res['gt'].append(curr_env.context_dict['qpos'][curr_env.cur_t])
                        res['target'].append(curr_env.target['qpos'])
                        # at time step 0, this is the initial state.
                        # At timestep 1, this is the predicted state at timestep 0 (which corresponds to 0)
                        pred_pose = curr_env.get_humanoid_qpos()
                        res['pred'].append(pred_pose)

                        if 'joints_gt' in curr_env.context_dict:
                            res["gt_jpos"].append(smpl_op_to_op(self.env.context_dict['joints_gt'][self.env.cur_t].copy()))
                            res["pred_jpos"].append(smpl_op_to_op(curr_env.get_wbody_pos().copy().reshape(24, 3)[MUJOCO_2_SMPL][curr_env.smpl_2op_submap]))
                        else:
                            res["gt_jpos"].append(self.env.gt_targets['wbpos'][self.env.cur_t].copy())
                            res["pred_jpos"].append(self.env.get_wbody_pos().copy())

                        # res["gt_jpos"].append(curr_env.gt_targets['wbpos'][curr_env.cur_t].copy())
                        # res["pred_jpos"].append(curr_env.get_wbody_pos().copy())
                        ###### When done, collect the last simulated state.

                        res = {k: np.vstack(v) for k, v in res.items()}
                        # print(info['percent'], context_dict['ar_qpos'].shape[1], loader.curr_key, np.mean(res['reward']))
                        res['percent'] = info['percent']
                        res['fail_safe'] = fail
                        res.update(compute_metrics(res, None))
                        return res

                    state = next_state



    def data_collect(self, num_jobs=10, num_samples=20, full_sample=False):
        cfg = self.cfg
        res_dicts = []
        data_collected = []
        with to_cpu(*self.sample_modules):
            with torch.no_grad():
                queue = multiprocessing.Queue()
                for i in range(num_jobs - 1):
                    worker_args = (queue, num_samples, full_sample)
                    worker = multiprocessing.Process(target=self.data_collect_worker, args=worker_args)
                    worker.start()
                res = self.data_collect_worker(None, num_samples, full_sample)
                data_collected += res
                for i in range(num_jobs - 1):
                    res = queue.get()
                    data_collected += res

        return data_collected

    def data_collect_worker(self, queue, num_samples=20, full_sample=False):

        curr_env = self.env

        with to_cpu(*self.sample_modules):
            with torch.no_grad():
                res = []
                loader = np.random.choice(self.train_data_loaders)
                self.set_mode('train')
                for i in range(num_samples):
                    if full_sample:
                        context_sample = loader.sample_seq(full_sample=full_sample, full_fr_num=True)
                    else:
                        context_sample = loader.sample_seq(fr_num=self.cfg.fr_num, full_fr_num=True)

                    context_sample = self.policy_net.init_context(context_sample, random_cam=True)
                    res.append({k: v.numpy() if torch.is_tensor(v) else v for k, v in context_sample.items()})

                if queue == None:
                    return res
                else:
                    queue.put(res)

    def sample_worker(self, pid, queue, min_batch_size):
        self.seed_worker(pid)
        memory = Memory()
        logger = self.logger_cls()
        self.policy_net.set_mode('test')
        self.env.set_mode('train')
        freq_dict = defaultdict(list)

        while logger.num_steps < min_batch_size:
            self.data_loader = np.random.choice(self.train_data_loaders)

            context_sample = self.data_loader.sample_seq(fr_num=self.cfg.fr_num)
            # should not try to fix the height during training!!!
            ar_context = self.policy_net.init_context(context_sample, random_cam=(not "cam" in context_sample))
            self.env.load_context(ar_context)
            state = self.env.reset()
            self.policy_net.reset()

            if self.running_state is not None:
                state = self.running_state(state)
            logger.start_episode(self.env)
            self.pre_episode()

            for t in range(10000):
                state_var = tensor(state).unsqueeze(0).double()
                trans_out = self.trans_policy(state_var)
                # mean_action = self.mean_action or self.env.np_random.binomial(
                #     1, 1 - self.noise_rate)

                mean_action = True
                action = self.policy_net.select_action(trans_out, mean_action)[0].numpy()

                action = int(action) if self.policy_net.type == 'discrete' else action.astype(np.float64)
                #################### ZL: Jank Code.... ####################
                # Gather GT data. Since this is before step, the gt needs to be advanced by 1. This corresponds to the next state,
                # as we collect the data after the step.
                humor_target = np.concatenate([self.env.context_dict[k][self.env.cur_t + 1].flatten() for k in self.env.agg_data_names])

                sim_humor_state = np.concatenate([self.env.cur_humor_state[k].numpy().flatten() for k in self.env.motion_prior.data_names])

                #################### ZL: Jank Code.... ####################

                next_state, env_reward, done, info = self.env.step(action)

                if self.running_state is not None:
                    next_state = self.running_state(next_state)
                # use custom or env reward
                if self.custom_reward is not None:
                    # Reward is not used.
                    c_reward, c_info = self.custom_reward(self.env, state, action, info)
                    reward = c_reward
                else:
                    c_reward, c_info = 0.0, np.array([0.0])
                    reward = env_reward

                # if flags.debug:
                #     np.set_printoptions(precision=4, suppress=1)
                #     print(c_reward, c_info)

                # add end reward
                if self.end_reward and info.get('end', False):
                    reward += self.env.end_reward
                # logging
                logger.step(self.env, env_reward, c_reward, c_info, info)

                mask = 0 if done else 1
                exp = 1 - mean_action
                self.push_memory(memory, state, action, mask, next_state, reward, exp, humor_target, sim_humor_state)

                if pid == 0 and self.render:
                    for _ in range(10):
                        self.env.render()

                if done:
                    freq_dict[self.data_loader.curr_key].append([info['percent'], self.data_loader.fr_start])
                    # print(self.data_loader.fr_start, self.data_loader.curr_key, info['percent'], self.env.cur_t)
                    break

                state = next_state

            logger.end_episode(self.env)
        logger.end_sampling()

        if queue is not None:
            queue.put([pid, memory, logger, freq_dict])
        else:
            return memory, logger, freq_dict

    def push_memory(self, memory, state, action, mask, next_state, reward, exp, humor_target, sim_humor_state):
        v_meta = np.array([self.data_loader.curr_take_ind, self.data_loader.fr_start, self.data_loader.fr_num])
        memory.push(state, action, mask, next_state, reward, exp, v_meta, humor_target, sim_humor_state)

    def optimize_policy(self, epoch):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        self.iter = epoch
        t0 = time.time()
        self.pre_epoch_update(epoch)

        self.cfg.lr = 5.e-5
        self.cfg.model_specs['weights']['l1_loss'] = 5
        self.cfg.model_specs['weights']['l1_loss_local'] = 0
        self.cfg.model_specs['weights']['loss_tcn'] = 1
        self.cfg.model_specs['weights']['prior_loss'] = 0.0001 if self.cfg.model_specs.get("use_prior", False) else 0
        self.cfg.model_specs['weights']['loss_2d'] = 0
        self.cfg.model_specs['weights']['loss_chamfer'] = 0
        self.cfg.policy_specs["num_step_update"] = 10
        self.cfg.policy_specs["rl_update"] = False
        cfg.policy_specs['min_batch_size'] = 5000
        if flags.debug:
            cfg.policy_specs['min_batch_size'] = 50
        cfg.save_n_epochs = 100
        cfg.eval_n_epochs = 100
        # cfg.policy_specs['min_batch_size'] = 500

        self.cfg.fr_num = 300 if self.iter < self.num_supervised else max(int(min(self.iter / 100, 1) * self.fr_num), self.cfg.data_specs.get("t_min", 30))

        if self.iter >= self.num_warmup:
            self.env.simulate = True
            self.cfg.model_specs['load_scene'] = self.load_scene
        else:
            self.env.simulate = False
            self.cfg.model_specs['load_scene'] = False
            # warm up should not load the scene......

        # cfg.policy_specs['min_batch_size'] = 50
        if self.cfg.lr != self.policy_net.optimizer.param_groups[0]['lr']:
            self.policy_net.setup_optimizers()

        batch, log = self.sample(cfg.policy_specs['min_batch_size'])

        if cfg.policy_specs['end_reward']:
            self.env.end_reward = log.avg_c_reward * cfg.policy_specs['gamma'] / (1 - cfg.policy_specs['gamma'])
        """update networks"""
        t1 = time.time()
        self.update_params(batch)
        t2 = time.time()
        info = {'log': log, 'T_sample': t1 - t0, 'T_update': t2 - t1, 'T_total': t2 - t0}

        if (self.iter + 1) % cfg.save_n_epochs == 0:
            self.save_checkpoint(self.iter)

        if (self.iter + 1) % cfg.eval_n_epochs == 0:
            log_eval = self.eval_policy("test")
            info['log_eval'] = log_eval

        if (self.iter + 1) % 5 == 0:
            self.save_curr()

        self.log_train(info)
        joblib.dump(self.freq_dict, osp.join(cfg.result_dir, "freq_dict.pt"))

    def update_params(self, batch):

        t0 = time.time()
        to_train(*self.update_modules)
        states = torch.from_numpy(batch.states).to(self.dtype).to(self.device)
        actions = torch.from_numpy(batch.actions).to(self.dtype).to(self.device)
        rewards = torch.from_numpy(batch.rewards).to(self.dtype).to(self.device)
        masks = torch.from_numpy(batch.masks).to(self.dtype).to(self.device)
        exps = torch.from_numpy(batch.exps).to(self.dtype).to(self.device)
        v_metas = torch.from_numpy(batch.v_metas).to(self.dtype).to(self.device)
        humor_target = torch.from_numpy(batch.humor_target).to(self.dtype).to(self.device)
        sim_humor_state = torch.from_numpy(batch.sim_humor_state).to(self.dtype).to(self.device)

        with to_test(*self.update_modules):
            with torch.no_grad():
                values = self.value_net(self.trans_value(states[:, :self.policy_net.state_dim]))

        seq_data = (masks, v_metas, rewards)
        self.policy_net.set_mode('train')
        self.policy_net.recrete_eps(seq_data)
        """get advantage estimation from the trajectories"""
        print("==================================================>")

        if self.cfg.policy_specs.get("rl_update", False):
            print("RL:")
            advantages, returns = estimate_advantages(rewards, masks, values, self.gamma, self.tau)
            self.update_policy(states, actions, returns, advantages, exps)

        if self.cfg.policy_specs.get("init_update", False) or self.cfg.policy_specs.get("step_update", False) or self.cfg.policy_specs.get("full_update", False):
            print("Supervised:")

        # if self.cfg.policy_specs.get("init_update", False):
        #     self.policy_net.update_init_supervised(self.cfg, self.data_loader, device=self.device, dtype=self.dtype, num_epoch=int(self.cfg.policy_specs.get("num_init_update", 5)))

        if self.cfg.policy_specs.get("step_update", False):

            self.policy_net.update_supervised(states, humor_target, sim_humor_state, seq_data, num_epoch=int(self.cfg.policy_specs.get("num_step_update", 10)))

        # if self.cfg.policy_specs.get("full_update", False):
        #     self.policy_net.train_full_supervised(self.cfg, self.data_loader, device=self.device, dtype=self.dtype, num_epoch=1, scheduled_sampling=0.3)

        # self.policy_net.step_lr()

        gc.collect()
        torch.cuda.empty_cache()

        return time.time() - t0
