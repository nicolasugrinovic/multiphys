import numpy as np
import copy
import socket
import torch
from collections import defaultdict

from metrics.tools import (
    local_align_joints,
    global_align_joints,
    first_align_joints,
)

hostname = socket.gethostname()

def list_dicts_to_stacked_dict(res):
    """for numpy arrays"""
    results = defaultdict(list)
    for this_res in res:
        for k, v in this_res.items():
            results[k].append(v)
    for k, v in results.items():
        results[k] = np.stack(v, axis=0)
    return results

def reorder_joints(gt_seq_joints, njts):
    # N = len(gt_seq_joints)
    N = 2
    gt_seq_joints_ = gt_seq_joints.reshape(N, -1, njts, 3) # (2, B, 24, 3)
    gt_p1 = gt_seq_joints_[0] # (B, 24, 3)
    gt_p2 = gt_seq_joints_[1] # (B, 24, 3)
    gt_seq_joints_ = torch.cat([gt_p1, gt_p2], dim=1) # (B, 2*24, 3)
    return gt_seq_joints_

def split_joints_per_person(pred, njts):
    pred_1 = pred[:, :njts]
    pred_2 = pred[:, njts:]
    pred_ = torch.stack([pred_1, pred_2], dim=1)
    return pred_

def compute_metrics_mp(res, converter=None, key=None, args=None, n=None, use_wbpos=False, use_smpl_gt=False, debug=False,
                       num_agents=2):
    """
    Computes metrics for multi-person data.
    :param res: dictionary containing the results for a single frame
    :param num_agents: number of agents in the scene
    """
    res = copy.deepcopy(res)
    res_dict = {}
    res = list_dicts_to_stacked_dict(res)
    jpos_pred = res["pred_jpos"]
    jpos_gt = res["gt_jpos"]
    b_res = jpos_pred.shape[1]
    b_gt = jpos_gt.shape[1]
    b_min = min(b_res, b_gt)
    jpos_pred = jpos_pred[:, :b_min]
    jpos_gt = jpos_gt[:, :b_min]
    jpos_pred = jpos_pred.reshape(-1, 24, 3)  # (2*B, 24, 3)
    jpos_gt = jpos_gt.reshape(-1, 24, 3)  # (2*B, 24, 3)

    if 'gt_jpos_vis' in res:
        vis = res['gt_jpos_vis'].astype(bool)
        vis = vis[0] # due to list_dicts_to_stacked_dict()
        jpos_pred = jpos_pred[:, vis]
        jpos_gt = jpos_gt[:, vis]

    njts = jpos_pred.shape[1]
    res_joints = torch.from_numpy(jpos_pred).float()
    gt_seq_joints = torch.from_numpy(jpos_gt).float()

    metric_names = ["ga_jmse", "fa_jmse", "pampjpe"]
    for name in metric_names:
        target = gt_seq_joints
        if name == "pampjpe":
            gt_seq_joints_ = reorder_joints(gt_seq_joints, njts) # (B, 2*24, 3)
            res_joints_ = reorder_joints(res_joints, njts) # (B, 2*24, 3)
            pred = local_align_joints(gt_seq_joints_, res_joints_) # (B, 2*24, 3)
            pred = split_joints_per_person(pred, njts) # (B, 2, 24, 3)
            target = split_joints_per_person(gt_seq_joints_, njts) # (B, 2, 24, 3)
        elif name == "ga_jmse":
            pred = global_align_joints(gt_seq_joints, res_joints) # (2*B, 24, 3)
        elif name == "fa_jmse":
            pred = first_align_joints(gt_seq_joints, res_joints) # (2*B, 24, 3)
        else:
            raise NotImplementedError
        # target and pred are: (2*B, 24, 3) for ga_jmse and fa_jmse, (B, 2, 24, 3) for pampjpe but the result does not change as dim=-1
        err = 1000 * torch.linalg.norm(target - pred, dim=-1).mean()
        res_dict[name] = err.item()

        # get errors per frame
        if name == "pampjpe":
            err_per_fr = 1000 * torch.linalg.norm(target - pred, dim=-1).mean(-1)  # (B, 2)
            res_dict['pampjpe_per_fr'] = err_per_fr
        elif name == "ga_jmse":
            target_ = target.reshape(num_agents, -1, njts, 3) # (2, B, 24, 3)
            pred_ = pred.reshape(num_agents, -1, njts, 3)
            err_per_fr = 1000 * torch.linalg.norm(target_ - pred_, dim=-1).mean(-1) # (2, B)
            res_dict['ga_jmse_per_fr'] = err_per_fr.T  # (B, 2)
        elif name == "fa_jmse":
            target_ = target.reshape(num_agents, -1, njts, 3) # (2, B, 24, 3)
            pred_ = pred.reshape(num_agents, -1, njts, 3) # (2, B, 24, 3)
            err_per_fr = 1000 * torch.linalg.norm(target_ - pred_, dim=-1).mean(-1)  # (2, B)
            res_dict['fa_jmse_per_fr'] = err_per_fr.T  # (B, 2)

        debug = False
        if debug:
            from utils.joints3d import joints_to_skel
            from tqdm import trange
            op = f"inspect_out/result_joints_mp/{args.data_name}/{name}/{args.model_type}-{args.exp_name}/{key}"

            get_jts = True
            rad = 0.015
            sphere_rad = 0.02
            fmt = 'smpl_red'
            if name=='pampjpe':
                pred_mp = pred.clone()
                target_mp = target.clone()
            else:
                pred_mp = pred.reshape(2, -1, 15, 3).permute(1, 0, 2, 3)
                target_mp = target.reshape(2, -1, 15, 3).permute(1, 0, 2, 3)

            pred_mp = pred_mp[::5]
            target_mp = target_mp[::5]
            for n in trange(len(pred_mp)):
                joints_to_skel(pred_mp[n], f"{op}/pred/skels_pred_{n:03d}.ply", format=fmt, radius=rad,
                               sphere_rad=sphere_rad, save_jts=get_jts)
                joints_to_skel(target_mp[n], f"{op}/gt/skels_gt_{n:03d}.ply", format=fmt, radius=rad,
                               sphere_rad=sphere_rad, save_jts=get_jts)


            pred_mp = pred_mp[::5]
            target_mp = target_mp[::5]
            for n in trange(len(pred_mp)):
                joints_to_skel(pred_mp[n, 0, None], f"{op}_p1/pred/skels_pred_{n:03d}.ply", format=fmt,
                               radius=rad,sphere_rad=sphere_rad, save_jts=get_jts)
                joints_to_skel(pred_mp[n, 1, None], f"{op}_p2/pred/skels_pred_{n:03d}.ply", format=fmt,
                               radius=rad,sphere_rad=sphere_rad, save_jts=get_jts)
                joints_to_skel(target_mp[n, 0, None], f"{op}_p1/gt/skels_gt_{n:03d}.ply", format=fmt,
                               radius=rad,sphere_rad=sphere_rad, save_jts=get_jts)
                joints_to_skel(target_mp[n, 1, None], f"{op}_p2/gt/skels_gt_{n:03d}.ply", format=fmt,
                               radius=rad,sphere_rad=sphere_rad, save_jts=get_jts)



    return res_dict
