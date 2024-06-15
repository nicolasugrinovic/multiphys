"""
Created by: Nicolas Ugrinovic
"""
import argparse
import os
import sys
sys.path.append(os.getcwd())
import numpy as np
from scipy.spatial.transform import Rotation as sRot
import joblib
import torch
from pathlib import Path
from utils.smpl import smpl_to_verts
from utils.misc import save_trimesh
from tqdm import tqdm
from utils.smpl_robot import from_qpos_to_verts_w_model
from metrics.prepare_slahmr_results import get_emb_robot_models
from utils.process_utils import load_humanoid
from pyquaternion import Quaternion as Q
from multiphys.process_data.preprocess.smpl_transforms import rot_and_correct_smplx_offset_full
from metrics.prepare_slahmr_results import get_emb_paths
from metrics.prepare import get_all_results
from utils.net_utils import get_hostname
hostname = get_hostname()
from multiphys.process_data.process.process import read_SLAHMR_data
from metrics.prepare import get_proc_data

torch.set_default_dtype(torch.float32)


np.set_printoptions(precision=4)
smplx_folder = '/home/nugrinovic/code/CVPR_2024/EmbodiedPose/data'

def get_paths():
    if "oriong" in hostname:
        SLA_ROOT = "/sailhome/nugrinov/code/CVPR_2024/slahmr_release/slahmr"
    else:
        SLA_ROOT = "/home/nugrinovic/code/CVPR_2024/slahmr_release/slahmr"
    VIDEOS_ROOT = f"{SLA_ROOT}/videos"
    return SLA_ROOT, VIDEOS_ROOT

def read_proc_data(args, PROC_ROOT, subj_name):
    sub_n = f"_{subj_name}" if subj_name != args.data_name else ""
    if args.data_name == 'chi3d' or args.data_name == 'hi4d':
        p1 = f"{PROC_ROOT}/{args.data_name}/thirdeye_clip_{args.data_name}{sub_n}_embodied_cam2w_p1_floor_phalp_initSla.pkl"
        p2 = f"{PROC_ROOT}/{args.data_name}/thirdeye_clip_{args.data_name}{sub_n}_embodied_cam2w_p2_floor_phalp_initSla.pkl"
    else:
        p1 = f"{PROC_ROOT}/{args.data_name}/{args.data_name}{sub_n}_p1_phalpBox_all_slaInit_slaCam.pkl"
        p2 = f"{PROC_ROOT}/{args.data_name}/{args.data_name}{sub_n}_p2_phalpBox_all_slaInit_slaCam.pkl"

    proc_data = [joblib.load(p1)]
    proc_data.append(joblib.load(p2))
    return proc_data


def build_simu_res_dict(args, seq_name, key_name, n, proc_this, all_results, mean_shape, models):
    debug = False
    
    kp_25 = proc_this['joints2d']
    B_kpts = len(kp_25)
    qpos_pred = all_results[seq_name][n]['pred']
    B_pose = len(qpos_pred)
    B = min(B_kpts, B_pose)

    betas = mean_shape[n].cpu().numpy()
    # this introduces an 180z degree rotation, need to compensate
    pose_aa, trans, _ = from_qpos_to_verts_w_model(qpos_pred, models[n], betas=betas[:10], ret_body_pose=True)
    if args.data_name=='expi':
        pose_aa_rot = pose_aa
        trans_rot = trans
    else:
        Tz180 = Q(axis=[0, 0, 1], angle=np.pi).transformation_matrix.astype(np.float32)
        pose_aa_rot, trans_rot = rot_and_correct_smplx_offset_full(pose_aa, trans, betas[None, :10], Tz180[None])

    cam = proc_this['cam']

    start, end = 0, B

    pose_mat = sRot.from_rotvec(pose_aa_rot[start:end].reshape(B * 24, 3)).as_matrix().reshape(B, 24, 3, 3)
    pose_body = pose_mat[:, 1:22]
    root_orient = pose_mat[:, 0:1]

    if debug:
        op = f"inspect_out/process_simu"
        Path(op).parent.mkdir(parents=True, exist_ok=True)

        verts, faces = smpl_to_verts(pose_aa_rot[0, None], trans_rot[0, None], betas=betas[None, :10])
        save_trimesh(verts[0, 0], faces, f"{op}/pose_aa_rot_{n}.ply")
        verts, faces = smpl_to_verts(pose_aa[0, None], np.zeros_like(trans[0, None]), betas=betas[None, :10])
        save_trimesh(verts[0, 0], faces, f"{op}/pose_aa_root_rel_{n}.ply")

        pose_aa_pr = proc_this['pose_aa']
        trans_pr = proc_this['trans']
        betas_pr = proc_this['betas']
        verts, faces = smpl_to_verts(pose_aa_pr[0, None], trans_pr[0, None], betas=betas_pr[None, :10])
        save_trimesh(verts[0, 0], faces, f"{op}/proc/pose_proc_{n}.ply")

    out_dict= {
        "joints2d": kp_25.copy(),  # (B, 25, 3)

        "pose_aa": pose_aa_rot.reshape(-1, 72)[start:end],  # (B, 72)
        "root_orient": root_orient.squeeze(),  # (B, 3, 3)
        "pose_body": pose_body[start:end],  # (B, 21, 3, 3)
        "trans": trans_rot.squeeze()[start:end],  # (B, 3)
        'betas': betas.squeeze(),  # (16, )

        "pose_6d": np.zeros([B, 24, 6]),
        "trans_vel": np.zeros([B, 1, 3]),  # (B, 3)
        "root_orient_vel": np.zeros([B, 1, 3]),
        "joints": np.zeros([B, 22, 3]),  # (B, 66)
        "joints_vel": np.zeros([B, 22, 3]),
        "seq_name": key_name,
        "gender": "neutral",

        "points3d": None,
        "cam": cam
    }
    return out_dict


class SampleDataProcessor(object):
    def __init__(
        self,
        args,
        name='simu_output_01',
        **kwargs,
    ):
        self.args = args
        self.data_name = args.data_name
        self.filter_seq = args.filter_seq
        self.exp_name = args.exp_name
        self.name = name
        # call methods
        self.set_data_paths()

    def read_data(self):
        pass
    def process(self):
        pass
    def save(self):
        pass

    def set_data_paths(self):
        # data_name = args.data_name
        # filter = args.filter
        path_, SEQS, ROOT, RES_ROOT, SLA_ROOT, PROC_ROOT = get_emb_paths(self.data_name, self.exp_name, sla_root=True)
        PROC_ROOT = f"{ROOT}/sample_data"
        self.ROOT = ROOT
        self.PROC_ROOT = PROC_ROOT
        self.RES_ROOT = RES_ROOT
        self.SEQS = SEQS
        self.SLA_ROOT = SLA_ROOT

    def get_results(self, subj_name):
        path = f"{self.RES_ROOT}/{self.args.data_name}/{self.args.exp_name}/{subj_name}"
        path = Path(path)
        print(f"Reading results from:\n {path}")
        all_results = get_all_results(path, remove_subj=True)
        return all_results

    def get_betas_from_results(self, proc_data, seq_name, smpl_robot, subj_name, betas_swapped=True):
        if betas_swapped:
            p_map = {0: 1, 1: 0}
        else:
            p_map = {0: 0, 1: 1}
        mean_shape = []
        for n in range(2):
            try:
                # swap order here because the default option in eval_scene_multi is swap_order=1
                try:
                    pred_betas = proc_data[p_map[n]][seq_name]['betas']
                except:
                    pred_betas = proc_data[p_map[n]][f"{subj_name}_{seq_name}"]['betas']
                mean_shape.append(pred_betas)
            except:
                print(f"no data for {seq_name}, this was not saved a processing due to errors!")
                return None
        try:
            mean_shape = np.stack(mean_shape)
        except:
            print(f"ERROR: {seq_name} probably got estimate for only one person!")
        try:
            mean_shape = torch.from_numpy(mean_shape)
            models = get_emb_robot_models(mean_shape, smpl_robot)
        except:
            print("ERROR: probably got estimate for only one person!")
            return None
        out = {
            'mean_shape': mean_shape,
            'models': models,
            'p_map': p_map,

        }
        return out

    def get_slahmr_results(self, subj_name, seq_name):
        if subj_name== 'hi4d':
            subj_name = '.'
        #### Taking smpl init from SLAHMR
        slahmr_dict = read_SLAHMR_data(self.SLA_ROOT, self.data_name, subj_name, seq_name)
        if slahmr_dict is None:
            tqdm.write(f"skipping - {subj_name} - {seq_name}")
            return None

        res_dict = slahmr_dict['res_dict']
        pose_aa = slahmr_dict['pose_aa']
        trans_orig = slahmr_dict['trans_orig']
        betas = slahmr_dict['betas']
        try:
            floor_plane = res_dict['floor_plane'].cpu()
            cam_R = res_dict['cam_R'].cpu()
            cam_t = res_dict['cam_t'].cpu()
            intrins = res_dict['intrins'].cpu()
            src_cam = res_dict['cameras']['src_cam'].cpu()
            
        except:
            floor_plane = None
            cam_R = None
            cam_t = None
            intrins = None
            src_cam = None

        out_data = {
            'pose_aa': pose_aa,
            'trans': trans_orig,
            'betas': betas,
            'floor_plane': floor_plane,
            'cam_R': cam_R,
            'cam_t': cam_t,
            'intrins': intrins,
            'src_cam': src_cam
        }

        return out_data

    def get_seqs(self):
        return self.SEQS

    def get_proc_sample_data(self, subj_name):
        # proc_data = read_proc_data(self.args, self.PROC_ROOT, subj_name)
        proc_data, path = get_proc_data(self.args, self.PROC_ROOT, subj_name)
        return proc_data, path

    def create_output_dict(self):
        pass

    def save_per_person_data(self, new_dict, subj_name, add_name='', outdir_name=None):
        if outdir_name is None:
            outdir_name = self.args.exp_name
        for n in range(2):
            if self.args.data_name == 'chi3d' or self.args.data_name == 'hi4d':
                output_dir = f"{self.PROC_ROOT}/{self.data_name}_slahmr/{self.name}/{outdir_name}"
            else:
                output_dir = f"{self.PROC_ROOT}/{self.data_name}/{self.name}/{outdir_name}"
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            subj_n = f"_{subj_name}" if self.data_name =='chi3d' or self.data_name=='expi' else ""
            data_n = self.data_name #+ "_slahmr" if self.data_name == 'chi3d' else self.data_name
            debug_n = "_debug" if self.args.debug else ""
            person_id_n = f"_p{n+1}"
            dyncam_n = "_dyncam" if self.args.dyncam else ""
            filt_n = f"_{self.args.filter_seq}" if self.args.filter_seq else "_all"
            out_fpath = Path(output_dir) / f"{data_n}{subj_n}{debug_n}{person_id_n}{dyncam_n}{filt_n}_slaInit_slaCam{add_name}.pkl"
            joblib.dump(new_dict[n], out_fpath)
            print(new_dict[n].keys())
            print(f"Saved at : {out_fpath}")




def process(args, debug=False):


    smpl_robot, humanoid, cc_cfg = load_humanoid()

    count = 0
    if args.debug:
        print("*** DEBUG MODE ***")
        print("*** DEBUG MODE ***")
    data_name = args.data_name
    filter = args.filter_seq

    path_, SEQS, ROOT, RES_ROOT = get_emb_paths(args.data_name, args.exp_name)
    PROC_ROOT = f"{ROOT}/sample_data"

    for subj_name in SEQS:
        print(f"Doing {subj_name}...")
        new_dict = [{}, {}]
        
        path = f"{RES_ROOT}/{args.data_name}/{args.exp_name}/{subj_name}"
        path = Path(path)
        # read proc data
        proc_data = read_proc_data(PROC_ROOT, subj_name)
        # read result files
        print(f"Reading results from:\n {path}")
        all_results = get_all_results(path)

        for nn, seq_name in enumerate(tqdm(all_results)):

            # seq_name = 's02_Grab_17'

            results = all_results[seq_name]
            p_map = {0: 1, 1: 0}
            mean_shape = []
            for n in range(2):
                try:
                    # swap order here because the default option in eval_scene_multi is swap_order=1
                    pred_betas = proc_data[p_map[n]][seq_name]['betas']
                    mean_shape.append(pred_betas)
                except:
                    print(f"no data for {seq_name}, this was not saved a processing due to errors!")
                    continue
            try:
                mean_shape = np.stack(mean_shape)
            except:
                print(f"ERROR: {seq_name} probably got estimate for only one person!")
            try:
                mean_shape = torch.from_numpy(mean_shape)
                models = get_emb_robot_models(mean_shape, smpl_robot)
            except:
                print("ERROR: probably got estimate for only one person!")
                continue
            # key_name = f"{subj_name}_{seq_name}" if data_name == 'chi3d' or data_name == 'expi' else seq_name
            key_name = seq_name
            # Format data for saving
            for n in range(len(results)):
                proc_this = proc_data[p_map[n]][seq_name]
                out_dict = build_simu_res_dict(args, seq_name, key_name, n, proc_this, all_results, mean_shape, models)
                new_dict[n][key_name] = out_dict

            count += 1
            for n in range(2):
                # new_dict[n][key_name]["video_path"] = str(vfile)
                if args.debug:
                    excluded = ["betas", "cam", "points3d", "gender", "seq_name"]
                    for k, v in new_dict[n][key_name].items():
                        if k in excluded:
                            continue
                        new_dict[n][key_name][k] = v[:31]

            if args.debug and count > 1:
                break

        print(f"processed {count} sequences")
        # for n in range(len(pose_aa[:2])):
        for n in range(2):
            output_dir = f"{args.output_dir}/simu_output_01/{args.exp_name}"
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            subj_n = f"_{subj_name}" if data_name =='chi3d' or data_name=='expi' else ""
            data_n = data_name + "_slahmr" if data_name == 'chi3d' else data_name
            debug_n = "_debug" if args.debug else ""
            person_id_n = f"_p{n+1}"
            dyncam_n = "_dyncam" if args.dyncam else ""
            filt_n = f"_{args.filter_seq}" if filter else "_all"
            out_fpath = Path(output_dir) / f"{data_n}{subj_n}{debug_n}{person_id_n}{dyncam_n}{filt_n}_slaInit_slaCam.pkl"
            joblib.dump(new_dict[n], out_fpath)
            print(new_dict[n].keys())
            print(f"Saved at : {out_fpath}")

    print("Done")
    if 0:
        out = smpl_to_verts(pose_aa[n], trans[n], betas=betas[n][:, :10], return_joints=True)
        (verts, joints), faces = out
        joints = joints[0, 0, :24].detach().numpy()
        # this is the same as phalp.ply so it is ok!
        out_path = f"inspect_out/phalp_process/{data_name}/sla_init_{n}.ply"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        save_trimesh(verts[0, 0], faces, out_path)


def main(args):
    process(args)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default='chi3d', choices=['viw', 'chi3d', 'hi4d', 'expi', 'shorts'])
    parser.add_argument("--vid_name", type=str, default=None)
    # parser.add_argument("--subject", type=str, default='.')
    parser.add_argument("--output_dir", default='sample_data/videos_wild')
    parser.add_argument('--debug', type=int, choices=[0, 1], default=0)
    parser.add_argument("--joints_source", type=str, choices=['hybrik', 'phalp'], default='phalp')
    parser.add_argument("--bbox_source", type=str, choices=['hybrik', 'phalp'], default='phalp')
    parser.add_argument('--dyncam', type=int, choices=[0, 1], default=0)
    parser.add_argument("--filter", type=str, default=None)
    parser.add_argument("--filter_actions", type=int, choices=[0, 1], default=0)
    parser.add_argument("--exp_name", type=str, default='slahmr_override', choices=['slahmr_override',
                                                                              'slahmr_override_loop4'])
    args = parser.parse_args()

    main(args)