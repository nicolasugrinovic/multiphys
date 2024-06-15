import numpy as np
from utils.misc import read_pickle
from utils.joints3d import joints_to_skel
from tqdm import tqdm
import socket
from utils.misc import save_mesh
from utils.smpl import local_bm

from metrics.prepare_slahmr_results import get_emb_paths
import joblib
from uhc.smpllib.smpl_eval import p_mpjpe
from utils.smpl import smpl_to_verts

hostname = socket.gethostname()


def make_relative(pred_p1):
    pred_p1 = pred_p1 - pred_p1[:, 0:1]
    return pred_p1

def inspect_joints(args, data, seq_name):
    gt_jpos_p1 = data[seq_name][0]['gt_jpos']
    gt_jpos_p2 = data[seq_name][1]['gt_jpos']
    pred_jpos_p1 = data[seq_name][0]['pred_jpos']
    pred_jpos_p2 = data[seq_name][1]['pred_jpos']

    pred_jpos_p1 = pred_jpos_p1[::5].reshape([-1, 24, 3])
    pred_jpos_p2 = pred_jpos_p2[::5].reshape([-1, 24, 3])
    gt_jpos_p1 = gt_jpos_p1[::5].reshape([-1, 24, 3])
    gt_jpos_p2 = gt_jpos_p2[::5].reshape([-1, 24, 3])

    for n in range(len(pred_jpos_p1)):
        pred_p1 = pred_jpos_p1[n]
        pred_p2 = pred_jpos_p2[n]
        target_p1 = gt_jpos_p1[n]
        target_p2 = gt_jpos_p2[n]
        try:
            vis = data[seq_name][0]['gt_jpos_vis'].astype(bool)
            format = 'smpl_red'
        except:
            vis = np.ones_like(target_p1[:, 0]).astype(bool)
            format = 'smpl'

        pred = np.stack([pred_p1, pred_p2])
        target = np.stack([target_p1, target_p2])
        pred = pred[:, vis]
        target = target[:, vis]

        op = f"inspect_out/result_joints/{args.data_name}/{args.model_type}-{args.exp_name}/{seq_name}"

        rel_n = ""
        get_jts = True
        if n == 0:
            print(f"saved to {op}")
        rad = 0.015
        sphere_rad = 0.02
        joints_to_skel(pred, f"{op}{rel_n}/pred/skels_pred_{n:03d}.ply", format=format, radius=rad,
                       sphere_rad=sphere_rad, save_jts=get_jts)
        joints_to_skel(target, f"{op}{rel_n}/gt/skels_gt_{n:03d}.ply", format=format, radius=rad, sphere_rad=sphere_rad,
                       save_jts=get_jts)


def inspect_verts(args, pred_verts_p1, pred_verts_p2, seq_name):
    # pred_verts_p1 = data[seq_name][0]['pred_vertices']
    # pred_verts_p2 = data[seq_name][1]['pred_vertices']

    pred_verts_p1 = pred_verts_p1[::5]
    pred_verts_p2 = pred_verts_p2[::5]

    for n in range(len(pred_verts_p1)):
        pred_p1 = pred_verts_p1[n]
        pred_p2 = pred_verts_p2[n]
        pred = np.stack([pred_p1, pred_p2])
        op = f"inspect_out/processed_vertices/{args.data_name}/{args.model_type}-{args.exp_name}/{seq_name}"
        if n == 0:
            print(f"saved to {op}")
        save_mesh(pred, local_bm.faces, f"{op}/verts_pred_{n:03d}.ply")


def get_verts_from_sdata(data_p1, seq_name):
    pose_aa = data_p1[seq_name]['pose_aa']
    trans = data_p1[seq_name]['trans']
    betas = data_p1[seq_name]['betas']
    verts_p1, faces = smpl_to_verts(pose_aa, trans, betas=betas[None])
    return verts_p1

def main(args):
    import torch

    path, SEQS, ROOT, RES_ROOT, sdata_path = get_emb_paths(args.data_name, args.exp_name, slahmr_overwrite=True, get_add=True)

    for subj_name in SEQS:
        fpath_p1 = sdata_path + f"/{args.data_name}_slahmr_{subj_name}_p1_phalpBox_all_slaInit_slaCam.pkl"
        fpath_p2 = sdata_path + f"/{args.data_name}_slahmr_{subj_name}_p2_phalpBox_all_slaInit_slaCam.pkl"

        print(f"reading {fpath_p1}")
        data_p1 = joblib.load(fpath_p1)
        print(f"reading {fpath_p2}")
        data_p2 = joblib.load(fpath_p2)

        for nn, seq_name in enumerate(tqdm(data_p1)):
            if nn > 1:
                continue

            verts_p1 = get_verts_from_sdata(data_p1, seq_name)
            verts_p2 = get_verts_from_sdata(data_p2, seq_name)
            # verts = torch.concat([verts_p1, verts_p2]).permute(1, 0, 2, 3)
            inspect_verts(args, verts_p1[0], verts_p2[0], seq_name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, choices=['chi3d', 'hi4d', 'expi'], default='chi3d')
    parser.add_argument('--model_type', type=str, choices=['slahmr', 'baseline'], default='baseline')
    parser.add_argument('--exp_name', type=str, choices=['normal_op', 'slahmr_override', 'overwrite_gt',
                                                         'slahmr_override_loop4'], default='normal_op')

    parser.add_argument('--rel_joints', type=int, default=0, choices=[0, 1])
    parser.add_argument('--pa_joints', type=int, default=0, choices=[0, 1])
    parser.add_argument('--type', type=str, choices=['joints', 'verts',], default='joints')
    args = parser.parse_args()

    main(args)
