import argparse
import os
import sys
sys.path.append(os.getcwd())
import numpy as np
from scipy.spatial.transform import Rotation as sRot
import torch
from pathlib import Path
from utils.smpl import smpl_to_verts
from utils.misc import save_trimesh
from utils.misc import read_pickle
from tqdm import tqdm
from pyquaternion import Quaternion as Q
from multiphys.process_data.preprocess.smpl_transforms import rot_and_correct_smplx_offset_full
from utils.net_utils import get_hostname
hostname = get_hostname()
from multiphys.processors import SampleDataProcessor

torch.set_default_dtype(torch.float32)


np.set_printoptions(precision=4)
smplx_folder = '/home/nugrinovic/code/CVPR_2024/EmbodiedPose/data'


def build_optim_res_dict(args, seq_name, key_name, n, proc_this, optim_data):
    debug = True

    kp_25 = proc_this['joints2d']
    betas_proc = proc_this['betas']
    B_kpts = len(kp_25)
    pose_aa = optim_data[seq_name]['pose_aa'][n]
    trans = optim_data[seq_name]['trans'][n]
    betas = optim_data[seq_name]['betas'][n]
    if args.overwrite_betas:
        betas = betas_proc
    betas = np.concatenate([betas, np.zeros([6], dtype=np.float32)], axis=0)

    if args.data_name=='expi' or args.data_name=='hi4d':
        pass
    else:
        print(f"WARNING: rotating pose_aa by 180 deg, data_name {args.data_name}")
        Tz180 = Q(axis=[0, 0, 1], angle=np.pi).transformation_matrix.astype(np.float32)
        pose_aa, trans = rot_and_correct_smplx_offset_full(pose_aa, trans, betas[None, :10], Tz180[None])

    B_pose = len(pose_aa)
    B = min(B_kpts, B_pose)
    cam = proc_this['cam']
    start, end = 0, B

    pose_mat = sRot.from_rotvec(pose_aa[start:end].reshape(B * 24, 3)).as_matrix().reshape(B, 24, 3, 3)
    pose_body = pose_mat[:, 1:22]
    root_orient = pose_mat[:, 0:1]

    if debug:
        op = f"inspect_out/process_optim/{args.data_name}/{seq_name}"
        Path(op).parent.mkdir(parents=True, exist_ok=True)
        verts, faces = smpl_to_verts(pose_aa[0, None], trans[0, None], betas=betas[None, :10])
        save_trimesh(verts[0, 0], faces, f"{op}/pose_aa_{n}.ply")
        verts, faces = smpl_to_verts(pose_aa[0, None], np.zeros_like(trans[0, None]), betas=betas[None, :10])
        save_trimesh(verts[0, 0], faces, f"{op}/pose_aa_root_rel_{n}.ply")

    out_dict = {
        "joints2d": kp_25.copy(),  # (B, 25, 3)

        "pose_aa": pose_aa.reshape(-1, 72)[start:end],  # (B, 72)
        "root_orient": root_orient.squeeze(),  # (B, 3, 3)
        "pose_body": pose_body[start:end],  # (B, 21, 3, 3)
        "trans": trans.squeeze()[start:end],  # (B, 3)
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


class OptimDataProcessor(SampleDataProcessor):
    def get_optim_data(self, subj_name):
        print(f"SUBDIR is : {self.args.sub_dir}")
        if self.args.sub_dir is None:
            sub_dir = ''
        else:
            sub_dir = self.args.sub_dir

        if self.args.data_name == 'chi3d' or self.args.data_name == 'hi4d':
            # output_dir = f"{args.output_dir}/{self.data_name}_slahmr/{self.name}/results/{subj_name}"
            output_dir = f"{args.output_dir}/{self.data_name}_slahmr/{self.name}/{sub_dir}/{self.args.exp_name}/{subj_name}"
        else:
            # output_dir = f"{args.output_dir}/{self.data_name}/{self.name}/results/{subj_name}"
            output_dir = f"{args.output_dir}/{self.data_name}/{self.name}/{sub_dir}/{self.args.exp_name}/{subj_name}"


        print(f"Reading OPTIM results from:\n {output_dir}")
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
        files = list(output_dir.glob(f"*.pkl"))
        print(f"Found {len(files)} files")
        all_results = {}
        # subj = path.stem
        # files = sorted((path.glob(f"{subj_name}_*/results.pkl")))
        for file in tqdm(files, disable=True):
            seq_name = file.stem
            try:
                data = read_pickle(file)
                # if '/expi/' in str(path):
                #     k, v = list(data.items())[0]
                #     data = {f"{subj}_{k}": v}
            except:
                print(f"does not exist {file}! ")
                continue
            all_results[seq_name] = data

        return all_results


def process(args, debug=False):
    """
    prepares the optimized data to be feed again to the simulation, saves results in sample_data dir
    """
    count = 0
    if args.debug:
        print("*** DEBUG MODE ***")
        print("*** DEBUG MODE ***")

    sample_data_processor = OptimDataProcessor(args, name=args.optim_res_dir)
    SEQS = sample_data_processor.get_seqs()

    for subj_name in SEQS:
        print(f"Doing {subj_name}...")
        new_dict = [{}, {}]

        # read proc data
        proc_data, proc_path = sample_data_processor.get_proc_sample_data(subj_name)
        optim_data = sample_data_processor.get_optim_data(subj_name)

        for nn, seq_name in enumerate(tqdm(optim_data)):
            if args.filter_seq is not None and args.filter_seq not in seq_name:
                print(f"*** WARNING: skipping {args.filter_seq} ***")
                continue
            # results = all_results[seq_name]
            # key_name = f"{subj_name}_{seq_name}" if data_name == 'chi3d' or data_name == 'expi' else seq_name
            tqdm.write(f"{seq_name}")
            key_name = seq_name
            # Format data for saving
            for n in range(2):
                proc_this = proc_data[n][seq_name]
                out_dict = build_optim_res_dict(args, seq_name, key_name, n, proc_this, optim_data)
                new_dict[n][key_name] = out_dict

            count += 1
            for n in range(2):
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
        sample_data_processor.save_per_person_data(new_dict, subj_name, outdir_name=args.out_dir_name)

    print("Done")


def main(args):
    process(args)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default='chi3d', choices=['viw', 'chi3d', 'hi4d', 'expi', 'shorts'])
    parser.add_argument("--vid_name", type=str, default=None)

    # parser.add_argument("--subject", type=str, default='.')
    parser.add_argument("--output_dir", default='sample_data')
    parser.add_argument('--debug', type=int, choices=[0, 1], default=0)
    parser.add_argument("--joints_source", type=str, choices=['hybrik', 'phalp'], default='phalp')
    parser.add_argument("--bbox_source", type=str, choices=['hybrik', 'phalp'], default='phalp')
    parser.add_argument('--dyncam', type=int, choices=[0, 1], default=0)
    parser.add_argument("--filter_seq", type=str, default=None)
    parser.add_argument("--filter_actions", type=int, choices=[0, 1], default=0)
    parser.add_argument("--exp_name", type=str, default='slahmr_override')
    parser.add_argument("--sub_dir", type=str, default=None, choices=['results_simple'])
    parser.add_argument("--optim_res_dir", type=str, default='optim_slahmr_output_naive')
    parser.add_argument("--overwrite_betas", type=int, choices=[0, 1], default=0)
    parser.add_argument("--out_dir_name", type=str, default=None)
    args = parser.parse_args()

    main(args)