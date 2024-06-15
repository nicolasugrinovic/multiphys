'''

Adapted from EmbodiedPose/agents/agent_embodied_pose.py
Modified By: Nicolas Ugrinovic


'''

import gc
import multiprocessing
import os
import os.path as osp
import sys
from collections import defaultdict

import joblib
from tqdm import tqdm

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

from multiphys.data_loaders import data_dict
from multiphys.envs import env_dict
from embodiedpose.models import policy_dict
from embodiedpose.core.reward_function import reward_func
from embodiedpose.core.trajbatch_humor import TrajBatchHumor
from embodiedpose.models.humor.utils.humor_mujoco import MUJOCO_2_SMPL

from PIL import Image
from utils.misc import save_img
from pathlib import Path

from utils.visu_tools import extract_video_frames

import mujoco_py
import cv2
from utils.misc import plot
from utils.video import make_video
from scipy.spatial.transform import Rotation as sRot
from utils.misc import read_image_PIL
from utils.misc import plot_skel_cv2

def rot_img(img, show=False):
    im = Image.fromarray(img)
    angle = 180
    out = im.rotate(angle)
    out_rot = np.asarray(out)
    if show:
        plot(out_rot)
    return out_rot

class AgentEmbodiedPoseMulti(AgentUHM):

    def __init__(self, cfg, dtype, device, mode="train", checkpoint_epoch=0):
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.mode = mode
        self.num_agents = cfg.num_agents

        self.global_start_fr = 0
        self.iter = checkpoint_epoch
        self.num_warmup = 300
        self.num_supervised = 5
        self.fr_num = self.cfg.fr_num
        self.setup_vars()
        self.setup_data_loader()
        # inits the KinPolicyHumorRes
        self.setup_policy()
        self.setup_env()
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
        debug = False
        if debug:
            fname = "inspect_out/xml_s/robot2_AgentEmbodiedPose_init_middle.xml"
            self.env.smpl_robot2.write_xml(fname)

        super(AgentUHM, self).__init__(
            env=self.env,
            dtype=dtype,
            device=device,
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
            data_sample = self.data_loader.sample_seq(fr_num=20, fr_start=self.global_start_fr)
            context_multi = []
            for sample in data_sample:
                context_sample = self.policy_net.init_context(sample, random_cam=True)
                context_multi.append(context_sample)
        self.cc_cfg = cfg.cc_cfg

        ######################################## Init Env ######################################
        self.env = env_dict[self.cfg.env_name](cfg, init_context=context_multi,
                                               cc_iter=cfg.policy_specs.get('cc_iter', -1),
                                               mode="train", agent=self)
        # here the agent is already defined and in a T-pose lying in the floor
        ##########################################################################################
        self.env.seed(cfg.seed)

    def setup_policy(self):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        data_sample = self.data_loader.sample_seq(fr_num=20, fr_start=self.global_start_fr)
        # todo multi: maybe modify later for two agent data?
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
                data_loader = data_dict[dataset_name](cfg, [test_file], multiproess=False)
                self.test_data_loaders.append(data_loader)

        self.data_loader = np.random.choice(self.train_data_loaders)

    @staticmethod
    def setup_curr_env_viewer_cam(curr_env):
        curr_env.viewer = mujoco_py.MjRenderContextOffscreen(curr_env.sim, offscreen=True, opengl_backend='egl')
        full_R, full_t = curr_env.agent.env.camera_params['full_R'], curr_env.agent.env.camera_params['full_t']
        distance = np.linalg.norm(full_t)
        pos_3d = -full_R.T.dot(full_t)
        # get rotation matrix in euler angles, in degrees
        rotation = sRot.from_matrix(full_R).as_euler("XYZ", degrees=True)
        curr_env.viewer.cam.distance = distance
        lookat = pos_3d  # + x_axis
        curr_env.viewer.cam.lookat[:] = lookat
        curr_env.viewer.cam.azimuth = 90 - rotation[2]  # +10
        curr_env.viewer.cam.elevation = 90 - rotation[0]

    def visu_in_simulation(self, renders, take_key, data_time, output_dir):
        out_joint_imgs = osp.join(output_dir, f"simu_results")
        print(f"Saving sim visu imgs to {out_joint_imgs}")
        Path(out_joint_imgs).mkdir(parents=True, exist_ok=True)
        if self.cfg.subject is None:
            sub_name = take_key.split("_")[0]
        else:
            sub_name = self.cfg.subject
        img_files = extract_video_frames(self.cfg.data_name, take_key, sub_name)

        joints2d_all = []
        pred_joints2d_all = []
        for n in range(self.env.num_agents):
            joints2d = self.data_loader.data_raw[n][take_key]["joints2d"]
            joints2d_all.append(joints2d[1:])
            pred_joints2d = self.env.pred_joints2d[n][1:]
            pred_joints2d_all.append(np.concatenate(pred_joints2d, 0))

        joints2d_all = np.stack(joints2d_all, 1)
        pred_joints2d_all = np.concatenate(pred_joints2d_all, 1)
        j2_len = len(pred_joints2d_all)

        for i, rend in enumerate(tqdm(renders[:j2_len])):
            file = img_files[i]
            img = read_image_PIL(file)
            j2d = joints2d_all[i]
            pred_j2d = pred_joints2d_all[i]
            img_w_kpts2d = plot_skel_cv2(img, j2d)
            if self.cfg.vis_pred_kpts:
                img_w_kpts2d = plot_skel_cv2(img_w_kpts2d, pred_j2d, type="emb", alpha=0.6)
            rend = np.concatenate([img_w_kpts2d, rend], axis=1)
            outp = Path(f"{out_joint_imgs}/{i:06d}.png")
            save_img(outp, rend)
        make_video(outp)

    def eval_seq(self, take_key, loader, data_time):
        curr_env = self.env
        with to_cpu(*self.sample_modules):
            with torch.no_grad():
                # res = defaultdict(list)
                res = [defaultdict(list), defaultdict(list)]
                self.policy_net.set_mode('test')
                curr_env.set_mode('test')
                context_sample_all = loader.get_sample_from_key(take_key=take_key, full_sample=True,
                                                                return_batch=True, dyncam=curr_env.agent.cfg.dyncam)
                ar_context_all = []
                for context_sample in context_sample_all:
                    ar_context = self.policy_net.init_context(context_sample, random_cam=(not "cam" in context_sample))
                    ar_context_all.append(ar_context)

                ########################################################################################################
                curr_env.load_context(ar_context_all)
                print("* Last load_context() *")
                ########################################################################################################
                # the simu is updated here!
                state = curr_env.reset()
                ########################################################################################################
                if self.running_state is not None:
                    state = self.running_state(state)
                fail = False
                self.setup_curr_env_viewer_cam(curr_env)
                ########################### Starts the FOR LOOP ########################################################
                renders = []
                for t in range(10000):

                    if self.cfg.visualizer=="sim_render":
                        for i in range(1, 5):
                            img = curr_env.sim.render(width=400, height=400, camera_name=f"camera{i}")
                            path = Path(f"inspect_out/sim_render/{self.cfg.data_name}/{self.cfg.seq_name}/{data_time}/sim_cam{i}/%07d.png" % t)
                            path.parent.mkdir(parents=True, exist_ok=True)
                            save_img(path, img)
                    
                    action_all = []
                    pred_qpos = curr_env.get_humanoid_qpos()  # shape: (76,)
                    pred_jpos = self.env.get_wbody_pos().copy()  # shape: (72,)
                    for n in range(self.env.num_agents):
                        gt_qpos = curr_env.context_dict[n]['qpos'][curr_env.cur_t] # shape: (76,)
                        res[n]['gt'].append(gt_qpos)
                        target_qpos = curr_env.target[n]['qpos'] # shape: (1, 76)
                        res[n]['target'].append(target_qpos)
                        ########################## predicted pose #######################
                        start = n * self.env.qpos_dim
                        end = (n + 1) * self.env.qpos_dim
                        res[n]['pred'].append(pred_qpos[start:end])
                        #################################################################
    
                        if 'joints_gt' in curr_env.context_dict[n]:
                            res[n]["gt_jpos"].append(smpl_op_to_op(self.env.context_dict[n]['joints_gt'][self.env.cur_t].copy()))
                            res[n]["pred_jpos"].append(smpl_op_to_op(curr_env.get_wbody_pos().copy().reshape(24, 3)[MUJOCO_2_SMPL][curr_env.smpl_2op_submap]))
                        else:
                            gt_jpos = self.env.gt_targets[n]['wbpos'][self.env.cur_t].copy() # shape: (72,)
                            res[n]["gt_jpos"].append(gt_jpos)
                            start = n * 72
                            end = (n + 1) * 72
                            pred_jpos_ = pred_jpos[start:end]
                            res[n]["pred_jpos"].append(pred_jpos_)
    
                        if self.cfg.model_specs.get("use_tcn", False):
                            # these are 14 joints in 3D with no translation
                            world_body_pos = self.env.pred_tcn[n]['world_body_pos'].copy()[None,] # shape (1, 14, 3)
                            res[n]['world_body_pos'].append(world_body_pos)
                            world_trans = self.env.pred_tcn[n]['world_trans'].copy()[None,] # shape (1, 1, 3)
                            res[n]['world_trans'].append(world_trans)
    
                        state_var = tensor(state[n]).unsqueeze(0).double() # shape: (1, 6455)
                        trans_out = self.trans_policy(state_var)
                        ############################ Kinematic policy #####################################
                        action = self.policy_net.select_action(trans_out, mean_action=True)[0].numpy()
                        action = int(action) if self.policy_net.type == 'discrete' else action.astype(np.float64)
                        action_all.append(action)

                    ############################ The state gets updated here ###########################
                    next_state, env_reward, done, info = curr_env.step(action_all, kin_override=False)
                    for n in range(self.env.num_agents):
                        if isinstance(ar_context_all[n]['cam'], list):
                            cam_params = ar_context_all[n]['cam'][curr_env.cur_t]
                            curr_env.hb[n].update_projection(cam_params, curr_env.smpl2op_map, MUJOCO_2_SMPL)

                    if self.cfg.visu_in_simu:
                        img_w = self.env.camera_params["img_w"]
                        img_h = self.env.camera_params["img_h"]
                        curr_env.viewer.render(width=img_w, height=img_h)
                        data = curr_env.viewer.read_pixels(width=img_w, height=img_h, depth=False)
                        data_v = cv2.flip(data, 0)
                        renders.append(data_v)
                    ####################################################################################

                    if t % 20 == 0:
                        if self.cfg.debug:
                            done = True
                        print(f"Step {t}: is done? {done}, Failed? {info['fail']}, percent {info['percent']}")
                        # just for debugging
                    if self.cfg.render:
                        curr_env.render()
                    if self.running_state is not None:
                        next_state = self.running_state(next_state)

                    if info['fail']:
                        print("Fail!", take_key)
                        fail = info['fail']

                    if done:
                        if self.cfg.visu_in_simu:
                            output_dir = self.cfg.output_dir_seq
                            self.visu_in_simulation(renders, take_key, data_time, output_dir)

                        pred_pose = curr_env.get_humanoid_qpos()
                        pred_jpos = self.env.get_wbody_pos().copy()
                        for n in range(self.env.num_agents):
                            ###### When done, collect the last simulated state.
                            res[n]['gt'].append(curr_env.context_dict[n]['qpos'][curr_env.cur_t])
                            res[n]['target'].append(curr_env.target[n]['qpos'])
                            start = n * self.env.qpos_dim
                            end = (n + 1) * self.env.qpos_dim
                            res[n]['pred'].append(pred_pose[start:end])
                            if 'joints_gt' in curr_env.context_dict[n]:
                                res[n]["gt_jpos"].append(smpl_op_to_op(self.env.context_dict[n]['joints_gt'][self.env.cur_t].copy()))
                                res[n]["pred_jpos"].append(smpl_op_to_op(curr_env.get_wbody_pos().copy().reshape(24, 3)[MUJOCO_2_SMPL][curr_env.smpl_2op_submap]))
                            else:
                                res[n]["gt_jpos"].append(self.env.gt_targets[n]['wbpos'][self.env.cur_t].copy())
                                start = n * 72
                                end = (n + 1) * 72
                                pred_jpos_ = pred_jpos[start:end]
                                res[n]["pred_jpos"].append(pred_jpos_)
                            ###### When done, collect the last simulated state.
                            res[n] = {k: np.vstack(v) for k, v in res[n].items()}
                            res[n]['percent'] = info['percent']
                            res[n]['fail_safe'] = fail
                            res[n].update(compute_metrics(res[n], None))
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
        if self.cfg.policy_specs.get("step_update", False):
            self.policy_net.update_supervised(states, humor_target, sim_humor_state, seq_data, num_epoch=int(self.cfg.policy_specs.get("num_step_update", 10)))

        gc.collect()
        torch.cuda.empty_cache()

        return time.time() - t0
