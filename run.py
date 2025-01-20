'''
Adapted from EmbodiedPose/scripts/eval_scene.py
Modified by: Nicolas Ugrinovic
'''

import argparse
import os
import os.path as osp
import sys
import mujoco_py
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.append(os.getcwd())
import torch
import numpy as np
from uhc.utils.flags import flags
from uhc.utils.copycat_visualizer import CopycatVisualizer
from multiphys.agents import agent_dict
from embodiedpose.utils.video_pose_config import Config
from scipy.spatial.transform import Rotation as sRot
import cv2
from pathlib import Path
import time
import math
from utils.smpl import from_qpos_to_smpl
from utils.misc import write_str_txt
from utils.misc import write_pickle
import yaml
from utils.visu_tools import visu_estimates
from metrics.metrics_mp import compute_metrics_mp
from utils.inference import get_datetime, print_metrics



class SceneVisulizer(CopycatVisualizer):

    def update_pose(self, cam_num=0):
        lim = 76
        # this should contain the 6 robots: 3 per agent and per robot it is 76 --> 76 * 6 = 456
        for n in range(self.agent.env.num_agents):
            # this should go:
            # 0-76 --> 76-152 --> 152-228 --> 228-304 --> 304-380 --> 380-456
            self.env_vis.data.qpos[n*(lim * 3):n*(lim * 3)+lim] = self.data[n]["pred"][self.fr] # (76,)
            self.env_vis.data.qpos[n*(lim * 3)+lim:n*(lim * 3)+(lim * 2)] = self.data[n]["gt"][self.fr] # (76,)
            self.env_vis.data.qpos[n*(lim * 3)+(lim * 2):n*(lim * 3)+(lim * 3)] = self.data[n]["target"][self.fr] # (76,)

        if (self.agent.cfg.render_rfc and self.agent.cc_cfg.residual_force and self.agent.cc_cfg.residual_force_mode == "explicit"):
            self.render_virtual_force(self.data["vf_world"][self.fr])

        if self.agent.cfg.hide_im:
            self.env_vis.data.qpos[2] = 100.0
        if self.agent.cfg.hide_expert:
            self.env_vis.data.qpos[lim * 2 + 2] += 8 # target
            self.env_vis.data.qpos[lim * 5 + 2] += 8 # target

        if self.agent.cfg.hide_gt:
            self.env_vis.data.qpos[lim * 1 + 2] += 8 # target
            self.env_vis.data.qpos[lim * 4 + 2] += 8 # target

        if self.agent.cfg.hide_pred:
            self.env_vis.data.qpos[lim * 0 + 2] = 8
            self.env_vis.data.qpos[lim * 3+ 2] = 8

        if self.agent.cfg.shift_expert:
            self.env_vis.data.qpos[lim] += 3

        # moves the "target" agent out of the visu
        if self.agent.cfg.shift_kin:
            self.env_vis.data.qpos[lim * 1 + 1] -= 2 # GT
            self.env_vis.data.qpos[lim * 4 + 1] -= 2 # GT
            self.env_vis.data.qpos[lim * 2 + 1] += 2 # target
            self.env_vis.data.qpos[lim * 5 + 1] += 2 # target


        if self.fr == 146:
            self.agent.env.context_dict

        if self.agent.cfg.focus:
            self.env_vis.viewer.cam.lookat[:2] = self.env_vis.data.qpos[:2]

        ####################################################################################
        # sets the viewer with the camera pose
        if self.fr == 1:
            full_R, full_t = self.agent.env.camera_params['full_R'], self.agent.env.camera_params['full_t']
            distance = np.linalg.norm(full_t)
            pos_3d = -full_R.T.dot(full_t)
            # get rotation matrix in euler angles, in degrees
            rotation = sRot.from_matrix(full_R).as_euler("XYZ", degrees=True)
            if cam_num == 0 or cam_num == 2 or cam_num == 3:
                lookat = pos_3d #this is a vector
                if self.agent.cfg.closer_cam:
                    if self.agent.cfg.data_name == 'chi3d':
                        div = 2
                    elif self.agent.cfg.data_name == 'hi4d':
                        div = 1.2
                    else:
                        div = 2
                    self.env_vis.viewer.cam.distance = distance / div  # this does seem to have and effect
                    delta_trans = np.zeros_like(pos_3d)
                    delta_trans[2] = 0.3
                    self.env_vis.viewer.cam.lookat[:] = 0.5 * lookat + delta_trans
                else:
                    self.env_vis.viewer.cam.distance = distance  # this is a float, not a vector
                    self.env_vis.viewer.cam.lookat[:] = lookat

                self.env_vis.viewer.cam.azimuth = 90 - rotation[2] #+10
                self.env_vis.viewer.cam.elevation = 90 - rotation[0]
            else:
                self.env_vis.viewer.cam.distance = 0.8 * distance # this actually has an effect
                delta_trans = np.zeros_like(pos_3d)
                if self.agent.cfg.data_name in ['chi3d', 'chi3d_slahmr', 'hi4d']:
                    # z
                    delta_trans[2] = 1.5 # this is the z axis
                    lookat = pos_3d.copy()
                    new_lookat = 0.5 * pos_3d + delta_trans
                    self.env_vis.viewer.cam.lookat[:] = new_lookat
                    self.env_vis.viewer.cam.azimuth = 90 - rotation[2]  # + 180
                    ang_diff = np.arccos(lookat / new_lookat)
                    self.env_vis.viewer.cam.elevation = 90 - rotation[0] - 0.7 * np.degrees(ang_diff[2])
                else:
                    delta_trans[1] = 1.5 # this is the y axis
                    # z
                    delta_trans[2] = 2 # this is the z axis
                    lookat = pos_3d.copy()
                    new_lookat = pos_3d + delta_trans
                    self.env_vis.viewer.cam.lookat[:] = 0.7 * new_lookat
                    self.env_vis.viewer.cam.azimuth = 90 - rotation[2] #+ 180
                    ang_diff = np.arccos(lookat / new_lookat)
                    self.env_vis.viewer.cam.elevation = 90 - rotation[0] - 0.7 * np.degrees(ang_diff[2])

        ####################################################################################
        self.env_vis.sim_forward()
        print(f"Current frame: {self.fr}", end='\r')

    def show_animation(self, mode="human", duration=60, cam_num=0):
        frames = []
        self.t = 0
        while True:
            if self.t >= math.floor(self.T):
                if not self.reverse:
                    if self.fr < self.num_fr - 1:
                        self.fr += 1
                        if mode == "image":
                            rend_img = self.render("image")
                            # resize rend img, too large kills the process
                            h, w, _ = rend_img.shape
                            if w==2048 and h==2048:
                                rend_img = cv2.resize(rend_img, (w//2, h//2))
                            frames.append(rend_img)
                            if 0:
                                from utils.misc import plot
                                plot(rend_img)
                                plot(frames[0])
                        if mode == "image" and self.fr >= duration-1:
                            return frames
                    elif self.repeat:
                        self.fr = 0
                elif self.reverse and self.fr > 0:
                    self.fr -= 1
                self.update_pose(cam_num)

                self.t = 0
            ######################################
            # this renders the simulation result and can be streamed to image files
            if mode == "human":
                rend_img = self.render("human")
                frames.append(rend_img)
            ######################################
            if not self.paused:
                self.t += 1


    @staticmethod
    def get_gen_keys(cfg, res_dir, pname):

        if cfg.data_name == 'prox':
            generated_files = list(Path(cfg.result_dir).glob("prox/*/*/*.mp4"))
        elif cfg.data_name == 'chi3d' or cfg.data_name == 'chi3d_slahmr':
            generated_files = list(Path(res_dir).glob("*/*/*/*.mp4"))
        elif cfg.data_name == 'hi4d':
            generated_files = list(Path(res_dir).glob("*/*/*/*.mp4"))
        else:
            generated_files = list(Path(res_dir).glob("*/*/*.mp4"))
        generated_files = sorted(generated_files)
        # #
        if cfg.data_name == 'prox':
            gen_keys = [f.parts[5] for f in generated_files if f"{pname}" in f.name]
        elif cfg.data_name == 'chi3d' or cfg.data_name == 'chi3d_slahmr':
            gen_keys = [f.parts[7] for f in generated_files]
            if gen_keys:
                assert len(gen_keys[0].split('_')) == 3, "bad path format for checking existing!"
        elif cfg.data_name == 'hi4d' or cfg.data_name == 'expi':
            gen_keys = [f.parts[7] for f in generated_files]
        else:
            gen_keys = [f.parts[4] for f in generated_files]

        return gen_keys


    def save_metrics(self, output_dir, out_joint_imgs, print_eval_all):
        print('Saving metrics...')
        eval_out_dir = Path(out_joint_imgs)
        eval_fname = str(eval_out_dir) + "/eval_metrics.txt"
        print(f"**Saving eval metrics to {eval_fname}")
        write_str_txt(print_eval_all, eval_fname)

        article_info = [self.agent.cfg.__dict__, ]
        with open(f"{output_dir}/config.yaml", 'w') as yamlfile:
            yaml.dump(article_info, yamlfile)
        cmd_fname = str(eval_out_dir) + "/cmd.txt"
        write_str_txt(cfg.in_cmd, cmd_fname)


    def generate_visu(self, take_key , eval_res, vis_xmpl):
        frames_all = []
        visu_cam_num = self.agent.cfg.num_cameras + 2
        for cam_num in range(visu_cam_num):
            self.env_vis.reload_sim_model(
                vis_xmpl,
                self.agent.cfg.viewer_type
            )
            self.image_path = osp.join(
                self.agent.cfg.output, str(take_key),
                f"{take_key}_{self.agent.cfg.id}_{self.agent.cfg.epoch}_%04d.png",
            )
            self.video_path = osp.join(
                self.agent.cfg.output, f"{take_key}_{self.agent.cfg.id}_{self.agent.cfg.epoch}_%01d.mp4",
            )
            self.data = eval_res
            self.setup_viewing_angle(self.agent.cfg.viewer_type)
            ########################################### for visu ###########################################
            self.num_fr = eval_res[0]["pred"].shape[0]  # + 1
            dur = self.agent.data_loader.data_raw[0][take_key]["pose_aa"].shape[0]  # + 1
            ################################### SHOW ANIMATION ########################################
            if cam_num == 0:
                # show pred
                self.agent.cfg.hide_expert = True
                self.agent.cfg.hide_gt = True
                self.agent.cfg.hide_pred = False
                self.agent.cfg.shift_kin = False
            elif cam_num == 1:
                # show pred
                self.agent.cfg.hide_expert = True
                self.agent.cfg.hide_gt = True
                self.agent.cfg.hide_pred = False
                self.agent.cfg.shift_kin = False
            elif cam_num == 2:
                # show the kin poses
                self.agent.cfg.hide_expert = False
                self.agent.cfg.hide_gt = True
                self.agent.cfg.hide_pred = True
                self.agent.cfg.shift_kin = False
            elif cam_num == 3:
                # show GT poses
                self.agent.cfg.hide_expert = True
                self.agent.cfg.hide_gt = False
                self.agent.cfg.hide_pred = True
                self.agent.cfg.shift_kin = False

            frames = self.show_animation(self.agent.cfg.viewer_type, duration=dur, cam_num=cam_num)
            ###########################################################################################
            frames = frames[1:]
            self.fr = 0
            self.num_fr = 0
            self.T_arr = [1, 2, 4, 6, 8, 10, 12, 15, 20, 30, 40, 50, 60]
            self.T = 12
            self.paused = False
            self.reverse = False
            self.repeat = False
            frames_this = np.stack(frames)
            frames_all.append(frames_this)
        frames = np.concatenate(frames_all, axis=2)
        return frames, cam_num

    def save_results(self, output_dir, results_save, results_3d, save_meshes=False):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        results_path = f"{output_dir}/results.pkl"
        write_pickle(results_save, results_path)
        print("*" * 50)
        print(f"Results saved at: {results_path}")
        print("*" * 50)

        if cfg.save_meshes:
            pred_verts = results_3d["pred_verts"]
            gt_verts = results_3d["gt_verts"]
            faces = results_3d["faces"]
            save_meshes(pred_verts, gt_verts, faces, output_dir)

    def get_results(self, eval_res):
        pred_qpos, results_all = [], []
        pred_verts, gt_verts = [[], []], [[], []]
        print_eval_all = ''
        for n in range(self.agent.env.num_agents):
            print(f"Agent {n} Mass:", mujoco_py.functions.mj_getTotalmass(self.agent.env.model),
                  f"Seq_len: {eval_res[n]['gt'].shape[0]}")
            if cfg.model_specs.get("use_tcn", False):
                self.body_pos = eval_res[n]['world_body_pos'] + eval_res[n]['world_trans']

            beta = self.agent.env.context_dict[n]["beta"]
            beta = torch.from_numpy(beta[0, :10])[None].float()
            pred_qpos.append(eval_res[n]['pred'])
            for p_qpos in pred_qpos[-1]:
                verts, faces = from_qpos_to_smpl(p_qpos, self.agent.env, betas=beta, agent_id=n)
                pred_verts[n].append(verts[0, 0].detach().numpy())

            pred_verts[n] = np.stack(pred_verts[n], axis=0)
            gt_qpos = eval_res[n]['gt']
            for g_qpos in gt_qpos:
                verts, faces = from_qpos_to_smpl(g_qpos, self.agent.env, betas=beta, agent_id=n)
                gt_verts[n].append(verts[0, 0].detach().numpy())
            gt_verts[n] = np.stack(gt_verts[n], axis=0)
            result, print_str_eval = print_metrics(eval_res[n])
            results_all.append(result)
            print_eval_all += print_str_eval + "\n"

        metric_res = compute_metrics_mp(eval_res)
        metric_res = {k: v for k, v in metric_res.items() if 'fr' not in k}
        print_str_eval = " \t ".join([f"{k}" for k, v in metric_res.items()])  # keys
        print_str_eval += "\n "
        print_str_eval += " \t ".join([f"{v:.3f}" for k, v in metric_res.items()])  # values
        print_eval_all += print_str_eval + "\n"

        print_str = " \t".join([f"{k}: {v:.3f}" for k, v in metric_res.items()])
        print("!!!Metrics computed against GT")
        print(print_str)

        results_3d = {
            "pred_verts": pred_verts,
            "gt_verts": gt_verts,
            "faces": faces,
        }

        return results_all, results_3d, print_eval_all

    def save_visu_estimates(self, take_key, output_dir, pname, cfg, agent, eval_res):
        # this creates another Viewer
        fname = "inspect_out/xml_s/robot_visu.xml"
        vis_xmpl = self.agent.env.smpl_robot[0].export_vis_string_self(num=3,
                                                                       smpl_robot=self.agent.env.smpl_robot[0],
                                                                       num_cones=0,
                                                                       fname=fname
                                                                       ).decode("utf-8")

        frames, cam_num = self.generate_visu(take_key, eval_res, vis_xmpl)
        visu_estimates(cfg, agent, take_key, frames, output_dir, cam_num, pname)
        self.reset_visu_params()

    def reset_visu_params(self):
        self.fr = 0
        self.num_fr = 0
        self.T_arr = [1, 2, 4, 6, 8, 10, 12, 15, 20, 30, 40, 50, 60]
        self.T = 12
        self.paused = False
        self.reverse = False
        self.repeat = False

    def data_generator(self):
        """
        This is the main function to generate the corrected estimates
        """
        data_time = get_datetime()
        results_save = {}
        if self.agent.cfg.mode != "disp_stats":
            for loader in self.agent.test_data_loaders:
                loader_keys = loader.data_keys
                print(loader_keys)
                for n_seq, take_key in enumerate(loader_keys):
                    print(f"~~{take_key}~~")
                    if cfg.filter is not None:
                        if take_key!=cfg.filter:
                            continue

                    res_dir = osp.join(cfg.result_dir, f"{cfg.data_name}/{cfg.name}")
                    pname = Path(cfg.data).stem.split('_')
                    pname = [c for c in pname if c.startswith('p')][0]
                    gen_keys = self.get_gen_keys(cfg, res_dir, pname)
                    if take_key in gen_keys and self.agent.cfg.skip_existing:
                        continue
                    print(f"Generating for {take_key} seqlen: {loader.get_sample_len_from_key(take_key)}")

                    context_sample_all = loader.get_sample_from_key(take_key, full_sample=True, return_batch=True)
                    condition = cfg.data_name in ['chi3d', 'chi3d_slahmr', 'expi', 'expi_slahmr']
                    subj = take_key.split("_")[0] if condition else '.'
                    # if 'hi4d' in cfg.data_name:
                    subj = "hi4d" if 'hi4d' in cfg.data_name else subj
                    output_dir = osp.join(cfg.result_dir, f"{cfg.data_name}/{cfg.name}/{subj}/{take_key}/{data_time}")
                    self.agent.cfg.output_dir_seq = output_dir

                    ar_context_all = []
                    for context_sample in context_sample_all:
                        random_cam = not 'cam' in context_sample
                        ar_context = self.agent.policy_net.init_context(context_sample, random_cam=random_cam)
                        ar_context_all.append(ar_context)
                    self.agent.env.load_context(ar_context_all)

                    ################ the PHYS-correction takes place here   ###################
                    start = time.time()
                    eval_res = self.agent.eval_seq(take_key, loader, data_time)
                    ############################################################################
                    if eval_res[0]['fail_safe']:
                        print(f"Simulation FAILED for {take_key}!!!")
                        agent.env.pred_joints2d = [[] for _ in range(self.agent.env.num_agents)]
                        continue

                    end = time.time()
                    inference_time = end - start
                    print(f"************ Inference time: {inference_time} ************")
                    # saves the xml file with the simulation model definition
                    self.agent.env.smpl_robot[0].write_xml("test.xml")

                    results_all, results_3d, print_eval_all = self.get_results(eval_res)
                    results_save[take_key] = results_all
                    self.save_results(output_dir, results_save, results_3d, save_meshes=cfg.save_meshes)
                    results_save = {}

                    # visualize result
                    self.save_visu_estimates(take_key, output_dir, pname, cfg, agent, eval_res)
                    self.save_metrics(output_dir, output_dir, print_eval_all)

                exit(0)
            yield eval_res
        else:
            yield None


if __name__ == "__main__":

    from utils.misc import plot
    import sys

    in_cmd = " ".join(sys.argv)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default=None)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--num_threads", type=int, default=30)
    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=-1)
    parser.add_argument("--show_noise", action="store_true", default=False)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no_log", action="store_true", default=False)
    parser.add_argument("--data", type=str, default="sample_data/thirdeye_anns_prox_overlap_no_clip.pkl")
    parser.add_argument("--mode", type=str, default="vis")
    parser.add_argument("--render_rfc", action="store_true", default=False)
    parser.add_argument("--render", action="store_true", default=False)
    parser.add_argument("--no_fail_safe", action="store_true", default=False)
    parser.add_argument("--focus", action="store_true", default=False)
    parser.add_argument("--output", type=str, default="test")
    parser.add_argument("--shift_expert", action="store_true", default=False)
    parser.add_argument("--shift_kin", action="store_false", default=True)
    parser.add_argument("--smplx", action="store_true", default=False)
    parser.add_argument("--hide_im", action="store_true", default=False)
    parser.add_argument("--filter_res", action="store_true", default=False)
    #
    parser.add_argument("--visualizer", type=str, default="viewer", choices=["viewer", "sim_render"])
    parser.add_argument("--viewer_type", type=str, default="image", choices=["human", "image"])
    parser.add_argument("--seq_name", type=str)
    parser.add_argument("--data_name", type=str, choices=['chi3d', 'prox', 'viw', 'hi4d', 'expi', 'shorts', 'mupots'])
    parser.add_argument("--ignore_align", type=int, choices=[0, 1], default=1)
    parser.add_argument("--ignore_fail", type=int, choices=[0, 1], default=0)
    parser.add_argument("--debug", type=int, choices=[0, 1], default=0)
    parser.add_argument("--two_agents", type=int, choices=[0, 1], default=0)
    parser.add_argument("--num_agents", type=int, choices=[1, 2], default=2)
    parser.add_argument("--dataset", type=str, choices=["multi_hum_pose"], default="multi_hum_pose")
    parser.add_argument("--visu_in_simu", type=int, choices=[0, 1], default=0)
    parser.add_argument("--overwrite_target_w_gt", type=int, choices=[0, 1], default=1)
    parser.add_argument("--save_meshes", type=int, choices=[0, 1], default=0)
    parser.add_argument("--name", type=str)
    parser.add_argument("--exclude_contacts", type=int, choices=[0, 1], default=0)
    parser.add_argument("--vis_pred_kpts", type=int, choices=[0, 1], default=0)
    parser.add_argument("--swap_order", type=int, choices=[0, 1], default=1)
    parser.add_argument("--hide_expert", type=int, choices=[0, 1], default=1)
    parser.add_argument("--hide_gt", type=int, choices=[0, 1], default=1)
    parser.add_argument("--hide_pred", action="store_true", default=False) # should always be false here!
    parser.add_argument("--num_cameras", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--closer_cam", type=int, choices=[0, 1], default=1)
    parser.add_argument("--skip_existing", type=int, choices=[0, 1], default=0)
    parser.add_argument("--dyncam", type=int, choices=[0, 1], default=0)
    parser.add_argument("--loops_uhc", type=int, default=1)
    parser.add_argument("--subject", type=str)
    parser.add_argument("--filter", type=str, default=None)
    args = parser.parse_args()

    # init config and parse config file
    cfg = Config(cfg_id=args.cfg, create_dirs=False)
    cfg.update(args)
    cfg.in_cmd = in_cmd
    flags.debug = args.debug
    cfg.no_log = True
    if args.no_fail_safe:
        cfg.fail_safe = False

    prox_path = cfg.data_specs['prox_path']
    cfg.output = osp.join(prox_path, "renderings/sceneplus", f"{cfg.id}")
    if cfg.mode == "vis":
        cfg.num_threads = 1
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
    print(f"Using: {device}")
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if cfg.smplx and cfg.robot_cfg["model"] == "smplh":
        cfg.robot_cfg["model"] = "smplx"

    cfg.data_specs["train_files_path"] = [(cfg.data, cfg.dataset)]
    cfg.data_specs["test_files_path"] = [(cfg.data, cfg.dataset)]

    if cfg.mode == "vis":
        cfg.num_threads = 1

    # create agent
    agent_class = agent_dict[cfg.agent_name]
    agent = agent_class(cfg=cfg, dtype=dtype, device=device, checkpoint_epoch=cfg.epoch, mode="test")
    vis_file = agent.env.smpl_robot[0].export_vis_string().decode("utf-8")
    # run pipeline
    vis = SceneVisulizer(vis_file, agent)
