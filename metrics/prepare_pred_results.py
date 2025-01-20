import numpy as np
from utils.misc import read_pickle
from utils.misc import write_pickle
from tqdm import tqdm
from pathlib import Path
from metrics.prepare_slahmr_results import parse_smpl_data
import socket
from utils.smpl_robot import from_qpos_to_verts_w_model
import joblib
import torch

from utils.process_utils import load_humanoid
from metrics.tools import match_w_hungarian
from pyquaternion import Quaternion as Q
from metrics.prepare_slahmr_results import parse_emb_gt_data
from metrics.prepare_slahmr_results import get_smpl_gt_data
from metrics.prepare_slahmr_results import smpl_to_qpose_torch
from metrics.prepare_slahmr_results import transform_coords_smpl_data
from metrics.prepare_slahmr_results import match_3d_joints
from metrics.prepare_slahmr_results import transform_pred_joints
from metrics.prepare_slahmr_results import get_emb_robot_models
from metrics.prepare_slahmr_results import rot_pose_180z
from metrics.prepare_slahmr_results import apply_transform
from metrics.prepare import parse_cam2world
from metrics.prepare import get_cam_from_preds
from metrics.prepare import get_all_results

from uhc.smpllib.torch_smpl_humanoid import Humanoid
from utils.smpl import smpl_to_verts

smpl_robot, humanoid, cc_cfg = load_humanoid()

hostname = socket.gethostname()


def rotate_180z(pred_jpos):
    Tz180 = Q(axis=[0, 0, 1], angle=np.pi).transformation_matrix.astype(np.float32)
    pred_jpos_r = (Tz180[None, :3, :3] @ pred_jpos.transpose(0, 2, 1)).transpose(0, 2, 1)
    pred_jpos_r = pred_jpos_r.reshape(-1, 72)
    return pred_jpos_r

def get_gt_data(args, subj_name):
    ROOT = "."
    subj_name = "." if args.data_name == 'hi4d' else subj_name

    DATA_ROOT = f"{ROOT}/data"
    sub_n = f"_{subj_name}" if subj_name != "." else ""
    chi3d_path = f"{DATA_ROOT}/{args.data_name}/{args.data_name}{sub_n}_embodied_cam2w_p1.pkl"
    print(f"\nReading GT PROC data from: {chi3d_path}")
    data_chi3d = read_pickle(chi3d_path)
    chi3d_path_2 = f"{DATA_ROOT}/{args.data_name}/{args.data_name}{sub_n}_embodied_cam2w_p2.pkl"
    data_chi3d_2 = read_pickle(chi3d_path_2)

    keys = sorted(data_chi3d.keys())
    print(f"\nKEYS from  GT PROC data are: {keys}\n")
    data_vars =  data_chi3d[keys[0]].keys()
    print(f"\nVARS from GT P1 PROC data are: {data_vars}")
    data_vars_2 =  data_chi3d_2[keys[0]].keys()
    print(f"VARS from GT P2 PROC data are: {data_vars_2}\n")

    return data_chi3d, data_chi3d_2, chi3d_path


def get_seq_gt_data(data_chi3d, data_chi3d_2, seq_name):
    gt_p1 = data_chi3d[seq_name]
    gt_p2 = data_chi3d_2[seq_name]
    gt_p1 = parse_smpl_data(gt_p1)
    gt_p2 = parse_smpl_data(gt_p2)
    gt_chi3d = [gt_p1, gt_p2]
    return gt_chi3d


def replace_slahmr_w_gt(all_results, all_results_ref, key):
    for p_id in range(2):
        qpos_pred = all_results[key][p_id]['pred']
        qpos_gt = all_results_ref[key][p_id]['pred']
        Bp = qpos_pred.shape[0]
        B = qpos_gt.shape[0]
        B = min(B, Bp)
        all_results[key][p_id]['gt'] = all_results_ref[key][p_id]['gt'][:B]
        all_results[key][p_id]['gt_jpos'] = all_results_ref[key][p_id]['gt_jpos'][:B]
        # rotate pred pose for 180 deg around z
        gt_jpos = all_results[key][p_id]['gt_jpos'].reshape(-1, 24, 3)
        pred_jpos = all_results[key][p_id]['pred_jpos'].reshape(-1, 24, 3)
        pred_jpos_r = rotate_180z(pred_jpos)
        all_results[key][p_id]['pred_jpos'] = pred_jpos_r
        all_results[key][p_id]['pred'] = all_results[key][p_id]['pred'][:B]
        all_results[key][p_id]['pred'] = all_results[key][p_id]['pred'][:B]

    debug = False
    if debug:
        from utils.misc import save_pointcloud
        op = "inspect_out/process_metrics/"
        save_pointcloud(pred_jpos[0], f"{op}/{key}/joints-rel/{args.exp_name}/pred_jpos_{p_id}.ply")
        save_pointcloud(pred_jpos_r[0].reshape(-1, 3), f"{op}/{key}/joints-rel/{args.exp_name}/pred_jpos_r_{p_id}.ply")
        save_pointcloud(gt_jpos[0], f"{op}/{key}/joints-rel/{args.exp_name}/gt_jpos_{p_id}.ply")

    return all_results


def order_persons(all_results, key):
    # order the GT as it does not have the same order as normal_op, with the first poses
    gt_jpos = np.stack([c['gt_jpos'][0].reshape(-1, 3) for c in all_results[key]])
    pred_jpos = np.stack([c['pred_jpos'][0].reshape(-1, 3) for c in all_results[key]])
    gt_jpos_rel = gt_jpos - gt_jpos[:, 0, None]
    pred_jpos_rel = pred_jpos - pred_jpos[:, 0, None]
    row_ind, col_ind = match_w_hungarian(gt_jpos_rel, pred_jpos_rel)

    gt_jpos = [all_results[key][i]['gt_jpos'] for i in col_ind]
    pred_jpos = [all_results[key][i]['pred_jpos'] for i in range(2)]
    gt = [all_results[key][i]['gt'] for i in col_ind]

    for i in range(2):
        all_results[key][i]['gt_jpos'] = gt_jpos[i]
        all_results[key][i]['gt'] = gt[i]

    debug = False
    if debug:
        from utils.misc import save_pointcloud
        op = "inspect_out/process_metrics/"
        gt_jpos = np.stack([c['gt_jpos'][0].reshape(-1, 3) for c in all_results[key]])
        pred_jpos = np.stack([c['pred_jpos'][0].reshape(-1, 3) for c in all_results[key]])

        save_pointcloud(gt_jpos[0], f"{op}/{key}/joints-rel_ord/{args.exp_name}/gt_jpos_0.ply")
        save_pointcloud(gt_jpos[1], f"{op}/{key}/joints-rel_ord/{args.exp_name}/gt_jpos_1.ply")
        save_pointcloud(pred_jpos[0], f"{op}/{key}/joints-rel_ord/{args.exp_name}/pred_jpos_0.ply")
        save_pointcloud(pred_jpos[1], f"{op}/{key}/joints-rel_ord/{args.exp_name}/pred_jpos_1.ply")
    
    return all_results
        

def prepare(args, debug=False):
    from collections import defaultdict
    from embodiedpose.models.humor.utils.humor_mujoco import SMPL_2_OP, MUJOCO_2_SMPL
    from metrics.prepare_slahmr_results import get_emb_paths
    from metrics.prepare import get_proc_data

    """
    preprocess CHI3D results files that have been divided into several small files,
    each one save in the results folder of the corresponding sequence
    """

    path_, SEQS, ROOT, RES_ROOT = get_emb_paths(args.data_name, args.exp_name)
    PROC_ROOT = f"{ROOT}/sample_data"

    for subj_name in SEQS:
        print(f"Doing {subj_name}...")
        path = f"{RES_ROOT}/{args.data_name}/{args.exp_name}/{subj_name}"
        path = Path(path)
        fname = f"../{subj_name}.pkl" if subj_name != "." else f"../{args.data_name}.pkl"
        outpath = path / fname

        print(f"going to save to: {outpath}")

        results_dict = defaultdict(list)
        results_dict_verts = defaultdict(list)

        data_chi3d_p1, data_chi3d_p2, gt_path = get_gt_data(args, subj_name)
        proc_data, proc_path = get_proc_data(args, PROC_ROOT, subj_name)
        all_results = get_all_results(path, is_heuristic=args.optim_heuristics)

        for n, seq_name in enumerate(tqdm(all_results)):
            # key = "s02_Grab_12"
            # gt_chi3d = get_seq_gt_data(data_chi3d_p1, data_chi3d_p2, key)

            gt_chi3d, Rt = parse_emb_gt_data(data_chi3d_p1, data_chi3d_p2, seq_name)
            smpl_dict_gt, smpl_gt_jts = get_smpl_gt_data(gt_chi3d)

            # read results
            jpos_pred_p1 = all_results[seq_name][0]['pred_jpos']
            jpos_pred_p2 = all_results[seq_name][1]['pred_jpos']
            jpos_pred = np.stack([jpos_pred_p1, jpos_pred_p2])

            qpos_pred_p1 = all_results[seq_name][0]['pred']
            qpos_pred_p2 = all_results[seq_name][1]['pred']
            qpos_pred = np.stack([qpos_pred_p1, qpos_pred_p2])

            seq_len = jpos_pred.shape[1]

            jpos_pred = torch.from_numpy(jpos_pred).float().cuda()
            jpos_pred = jpos_pred.reshape(2, -1, 24, 3)
            jpos_pred = jpos_pred[:, :, MUJOCO_2_SMPL]

            # transform first to cam coords to match slahmr space
            try:
                # proc_data is data processed for embpose, includes slahmr transform
                cam_dict = get_cam_from_preds(proc_data, seq_name)
            except:
                print(f"ERROR: no cam data for {seq_name}! skipping...")

            # this is from slahmr preds and Rt is the GT camera from each dataset
            world2cam = parse_cam2world(cam_dict)

            # if it is overwrite with SLAHMR then the transform is different, a 180 rotation over z is introduced, why?
            rot_180z = False if args.exp_name == 'normal_op' else True
            if rot_180z:
                jpos_pred_orig = jpos_pred.clone()
                jpos_pred = rot_pose_180z(jpos_pred.reshape(-1, 24, 3)).reshape(2, -1, 24, 3)

            # if I use directly the SMPL jts then I don't need to rotate them
            j_pred_cam = transform_pred_joints(jpos_pred, world2cam, rot_180z=False)

            if args.data_name == 'hi4d' or args.data_name == 'expi':
                # for hi4d the GT cam is world2cam, need to invert. for chi3d, hoeever, it is already cam2world
                cam2world = torch.linalg.inv(Rt)
            else:
                cam2world = Rt
            j_pred_world = transform_pred_joints(j_pred_cam, cam2world, rot_180z=False)

            if args.data_name == 'hi4d':
                Tx90 = Q(axis=[1, 0, 0], angle=np.pi / 2).transformation_matrix.astype(np.float32)
                Tz180 = Q(axis=[0, 0, 1], angle=np.pi).transformation_matrix.astype(np.float32)
                Tr = Tz180[None] @ Tx90[None]
                Tr = torch.from_numpy(Tr).float().cuda()
                j_pred_world = apply_transform(j_pred_world.reshape(-1, 24, 3), Tr).reshape(2, -1, 24, 3)

            exps_to_rot = ['simu01', 'optim', 'optim_naive', 'loop', 'slahmr_override', 'normal_op']
            exps_to_exclude = ['post_optim']
            contains = any([c in args.exp_name for c in exps_to_rot])
            exc = not any([c in args.exp_name for c in exps_to_exclude])
            rot_str = ""
            if (args.data_name == 'hi4d' or args.data_name == 'chi3d') and contains and exc:
                rot_str = f"** Add ROTATION of 180z! {args.exp_name} preds..."
                Tz180 = Q(axis=[0, 0, 1], angle=np.pi).transformation_matrix.astype(np.float32)
                Tr = Tz180[None]
                Tr = torch.from_numpy(Tr).float().cuda()
                j_pred_world = apply_transform(j_pred_world.reshape(-1, 24, 3), Tr).reshape(2, -1, 24, 3)

            # swap order here because the default option in eval_scene_multi is swap_order=1
            # if args.exp_name == 'normal_op' and args.data_name=='chi3d':
            #     print("WARNING: using normal order for BETAS!")
            #     p_map = {0: 0, 1: 1}
            # else:
            p_map = {0: 1, 1: 0}
            mean_shape = []
            for p_id in range(2):
                try:
                    # swap order here because the default option in eval_scene_multi is swap_order=1
                    pred_betas = proc_data[p_map[p_id]][seq_name]['betas']
                    mean_shape.append(pred_betas)
                except:
                    print(f"no data for {seq_name}, this was not saved a processing due to errors!")
                    continue
            try:
                mean_shape = np.stack(mean_shape)
            except:
                print(f"ERROR: {seq_name} probably got estimate for only one person!")

            tqdm.write(f"{args.exp_name} - {subj_name} - {seq_name} - {rot_str}")

            if debug:
                from utils.joints3d import joints_to_skel
                op = f"inspect_out/prepare_preds/{args.data_name}/{args.exp_name}/{seq_name}"
                # joints_to_skel(jpos_pred[:, 0].cpu(), f"{op}/skels_pred_{n:03d}.ply", format='smpl', radius=0.015)
                # joints_to_skel(j_pred_cam[:, 0].cpu(), f"{op}/skels_cam_{n:03d}.ply", format='smpl', radius=0.015)
                joints_to_skel(j_pred_world[:, 0].cpu(), f"{op}/skels_world_{n:03d}.ply", format='smpl', radius=0.015)
                joints_to_skel(smpl_gt_jts[:, 0].cpu(), f"{op}/skels_smplgt_{n:03d}.ply", format='smpl', radius=0.015)
                joints_to_skel(j_pred_world[:, -1].cpu(), f"{op}/skels_world_{n:03d}.ply", format='smpl', radius=0.015)
                joints_to_skel(smpl_gt_jts[:, -1].cpu(), f"{op}/skels_smplgt_{n:03d}.ply", format='smpl', radius=0.015)

            # match pred poses with GT poses, they are not in the same order
            # match j_gts from embdata  w/ slahmr preds in relative 3D space
            # row_ind, col_ind = match_3d_joints(j_pred_rot, j_gts)
            # match smpl to slahmr
            row_ind_smpl, col_ind_smpl = match_3d_joints(j_pred_world, smpl_gt_jts)
            seq_len_min = min(smpl_gt_jts.shape[1], seq_len)

            if row_ind_smpl is None:
                print(f"matching failed for {seq_name}! skipping...")
                continue

            try:
                mean_shape = torch.from_numpy(mean_shape)
                models = get_emb_robot_models(mean_shape, smpl_robot)
            except:
                print("ERROR: probably got estimate for only one person!")
                continue

            device = torch.device("cuda:0")

            pose_aa = []
            trans = []
            for p_id in range(2):
                model = models[p_id]
                shape = mean_shape[p_id]
                pose_aa_tr, trans_tr, body_pos = from_qpos_to_verts_w_model(qpos_pred[p_id], model, betas=shape[:10],
                                                                            ret_body_pose=True)
                pose_ = torch.tensor(pose_aa_tr).float()
                trans_ = torch.tensor(trans_tr).float()
                pose_aa.append(pose_)
                trans.append(trans_)
            pose_aa = torch.stack(pose_aa)
            trans = torch.stack(trans)
            mean_shape = mean_shape.float()

            if debug:
                from utils.misc import save_mesh
                from utils.misc import save_pointcloud
                op = f"inspect_out/prepare_res_meshes/{args.data_name}/{args.exp_name}/{seq_name}"
                # op = f"inspect_out/prepare_res_meshes_alt/{args.data_name}/{args.exp_name}/{seq_name}"
                idx = 0
                verts, faces = smpl_to_verts(pose_aa[:, idx], trans[:, idx], betas=mean_shape[:, :10])
                save_mesh(verts[0], faces, f"{op}/pred_pose_{idx:03d}.ply")
                trans_ = torch.zeros_like(trans)
                verts, faces = smpl_to_verts(pose_aa[:, idx], trans_[:, idx], betas=mean_shape[:, :10])
                save_mesh(verts[0], faces, f"{op}/pred_pose_{idx:03d}_root.ply")

                root1 = qpos_pred[0, 0, None, :3]
                root2 = qpos_pred[1, 0, None, :3]
                save_pointcloud(root1, f"{op}/pred_root_p1.ply")
                save_pointcloud(root2, f"{op}/pred_root_p2.ply")


            # do matching to get each correct kpts, copy slahmr code
            for p_id in range(len(jpos_pred[:2])):
                # NOTE: for some reason we have to rotate the joints and SMPL 180 over z-axis --> only when using gt from embpose.
                # these are now directly from smpl
                gt_jts = smpl_gt_jts[col_ind_smpl[p_id]]
                # pose_aa_w_rot_i, trans_w_rot_i, betas = transform_coords_smpl_data(p_id, pose_aa, trans, mean_shape, Rt, device)
                qpos_rot = smpl_to_qpose_torch(pose_aa[p_id].cuda(), models[p_id], trans=trans[p_id].cuda(), count_offset=False)
                qpos_rot = qpos_rot.cpu().numpy()

                gt_data = {}
                gt_data_verts = {}
                # qpos use only to compute the root metric
                # gt_data['gt'] = emb_data[col_ind_smpl[p_id]]['gt'][:seq_len_min] #.cpu().numpy()

                # BEWARE: dummy value as this metric is not important for now
                gt_data['gt'] = qpos_rot[:seq_len_min]
                gt_data['pred'] = qpos_rot[:seq_len_min]
                # joints these are used to compute MPJPE
                gt_jts_ = gt_jts[:seq_len_min].cpu().numpy()
                gt_data['gt_jpos'] = gt_jts_.reshape(-1, 72)
                # gt_data['pred_jpos'] = jpos_pred_world_rot[:seq_len_min].cpu().numpy()
                j_pred_r = j_pred_world[p_id, :seq_len_min].cpu().numpy()
                gt_data['pred_jpos'] = j_pred_r.reshape(-1, 72)
                # other
                gt_data['percent'] = 1.0
                gt_data['fail_safe'] = False

                # convert slahrm prediction to wbpos/jpos
                humanoid_fk = Humanoid(model=model)
                qpos_rot_t = torch.from_numpy(qpos_rot).float()
                fk_result = humanoid_fk.qpos_fk(qpos_rot_t, to_numpy=False)
                pred_wbpos = fk_result['wbpos'].numpy()
                # pred_wbpos = pred_wbpos.reshape(-1, 24, 3)
                gt_data['pred_jpos_wbpos'] = pred_wbpos

                smpl_dict = {
                    'pose_aa': pose_aa[p_id].float(),
                    'trans': trans[p_id].float(),
                    'betas': mean_shape[p_id, None].float(),
                }

                smpl_dict_gt = {
                    'pose_aa': gt_chi3d[col_ind_smpl[p_id]]['pose_aa'].float(),
                    'trans': gt_chi3d[col_ind_smpl[p_id]]['trans'].float(),
                    'betas': gt_chi3d[col_ind_smpl[p_id]]['shape'][0, None].float(),
                }
                gt_data['smpl_pred'] = smpl_dict
                gt_data['smpl_gt'] = smpl_dict_gt
                gt_data['gt_jpos_smpl'] = smpl_gt_jts[col_ind_smpl[p_id]][:seq_len_min]
                #
                results_dict[seq_name].append(gt_data)

                if args.save_verts:
                    verts, faces = smpl_to_verts(pose_aa[p_id], trans[p_id], betas=mean_shape[p_id, None, :10])
                    gt_data_verts['pred_vertices'] = verts[0].cpu().numpy()
                    #
                    results_dict_verts[seq_name].append(gt_data_verts)

            if debug:
                from utils.joints3d import joints_to_skel
                op = f"inspect_out/prepare_ord/{args.data_name}/{args.exp_name}/{seq_name}"
                pred_jpos_p1 = results_dict[seq_name][0]['pred_jpos']
                pred_jpos_p1 = pred_jpos_p1.reshape(-1, 24, 3)
                pred_jpos_p2 = results_dict[seq_name][1]['pred_jpos']
                pred_jpos_p2 = pred_jpos_p2.reshape(-1, 24, 3)

                gt_jpos_p1 = results_dict[seq_name][0]['gt_jpos']
                gt_jpos_p1 = gt_jpos_p1.reshape(-1, 24, 3)
                gt_jpos_p2 = results_dict[seq_name][1]['gt_jpos']
                gt_jpos_p2 = gt_jpos_p2.reshape(-1, 24, 3)

                joints_to_skel(pred_jpos_p1[0, None], f"{op}/pred_jpos_{n:03d}.ply", format='smpl', radius=0.015)
                joints_to_skel(gt_jpos_p1[0, None], f"{op}/gt_jpos_{n:03d}.ply", format='smpl', radius=0.015)
                joints_to_skel(pred_jpos_p2[0, None], f"{op}/pred_jpos_2_{n:03d}.ply", format='smpl', radius=0.015)
                joints_to_skel(gt_jpos_p2[0, None], f"{op}/gt_jpos_2_{n:03d}.ply", format='smpl', radius=0.015)

            # save the results

        write_pickle(results_dict, outpath)
        print(f"file saved to {outpath}")
        if args.save_verts:
            outpath_verts = str(outpath).replace(".pkl", "_verts.pkl")
            write_pickle(results_dict_verts, outpath_verts)
            print(f"file saved to {outpath}")


def main(args):
    from metrics.prepare_slahmr_results import prepare_slahmr
    if args.do_all:

        EXPERIMENTS = [
                        'normal_op',
                       'slahmr_override',
                       'slahmr_override_loop2',
                       'slahmr_override_loop3',
                       'slahmr_override_loop4',
                       'slahmr_override_loop5',
                       ]

        for exp in EXPERIMENTS:
            print(f"EXPERIMENT: {exp}...")
            args.exp_name = exp
            prepare(args, debug=False)
            # if exp== 'normal_op':
            #     prepare_slahmr(args)
    else:
        prepare(args, debug=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", type=int, choices=[0, 1], default=0)
    parser.add_argument("--debug", type=int, choices=[0, 1], default=0)
    parser.add_argument("--do_all", type=int, choices=[0, 1], default=0)
    # parser.add_argument("--subject", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default='normal_op')
    parser.add_argument("--vid_name", type=str, default='3_results_w_2d_p1_cat_sla')
    parser.add_argument("--data_name", type=str, default='chi3d', choices=['chi3d', 'hi4d', 'expi'])
    parser.add_argument('--filter_seq', type=str, default=None)
    parser.add_argument("--save_verts", type=int, choices=[0, 1], default=0)
    parser.add_argument("--optim_heuristics", type=int, choices=[0, 1], default=0)
    args = parser.parse_args()

    main(args)
