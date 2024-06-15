""" this file comes from embodiedpose/data_process/process_humor_split.py"""

import argparse
import os
import os.path as osp
import sys
sys.path.append(os.getcwd())
import joblib
import numpy as np
import torch
from tqdm import tqdm
from utils.misc import read_pickle
from pathlib import Path
from utils.pyquaternion import Quaternion as Q
from math import pi
from utils.smpl import smpl_to_verts
from utils.misc import save_trimesh
from utils.misc import read_image_PIL
import trimesh
from constants import *
from utils.process_utils import smpl_2_entry
from multiphys.process_data.process.process import prepare_multiperson_smpl_data
from multiphys.process_data.process.process import assemble_camera
from multiphys.process_data.process.process import slahmr_to_world
from multiphys.process_data.process.process import get_phalp_matched_kpts_simple_all
from multiphys.process_data.process.process import read_SLAHMR_data
from multiphys.process_data.process.data_reading import read_PHALP_data
from scipy.optimize import linear_sum_assignment


# smpl_robot, humanoid, cc_cfg = load_humanoid()
Tx = Q(axis=[0, 0, 1], angle=-pi / 2.).transformation_matrix.astype(np.float32)
Tx = torch.tensor(Tx[:3, :3])
np.random.seed(1)


def print_keys(motion_files):
    data4stats = read_pickle(motion_files[0])
    for k, v in data4stats.items():
        print(f"'{k}': {v['trans'].shape[0]},")
    # camera_pose = data["cam_50591643"]


def save_file(args, out_path, data_res, person_id):
    debug_name = "_debug" if args.debug else ""
    floor_name = "_floor" if args.to_floor else ""
    smooth_name = "_smooth2d" if args.smooth_2d else ""
    masked_name = "_masked" if args.mask_2d else ""
    jts_name = f"_{args.joints_source}"
    person_id_name = f"_p{person_id}"
    subset_name = "_subset" if args.subset else ""
    init_sla_n = "_initSla" if args.init_slahmr else ""
    output_file_name = osp.join(out_path, f'thirdeye_clip_{args.fname}{person_id_name}{debug_name}'
                                          f'{floor_name}{smooth_name}'
                                          f'{masked_name}{jts_name}{subset_name}{init_sla_n}.pkl')
    print(output_file_name, len(data_res))
    Path(output_file_name).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(data_res, open(output_file_name, "wb"))


def trans_to_floor(smpl_dict):
    debug=False
    pose = smpl_dict['pose']
    trans = smpl_dict['trans']
    betas = smpl_dict['shape']
    betas = torch.from_numpy(betas).float()  # .cuda()
    verts, faces = smpl_to_verts(pose, trans, betas=betas[0, None, :10])
    mesh = trimesh.Trimesh(vertices=verts[0, 0].detach().cpu().numpy(), faces=faces)
    bbox = mesh.bounding_box.bounds
    min_xyz = bbox[0]
    trans_floor_z = trans[:, 2] - min_xyz[None, 2]
    new_trans = np.concatenate([trans[:, :2], trans_floor_z[:, None]], 1)
    if debug:
        verts, faces = smpl_to_verts(pose[0, None], new_trans[0, None])
        save_trimesh(verts[0, 0], faces, f"inspect_out/chi3d/meshes/process/floor{person_name}.ply")

    return new_trans


def get_smpl_empty_dict(seq_length):
    gender = 'neutral'
    smpl_dict = {'pose': np.zeros((seq_length, 72)), 'trans': np.zeros((seq_length, 3)),
                 'shape': np.zeros((seq_length, 16)), 'joints2d': np.zeros((seq_length, 25, 3)),
                 'gender': gender}
    return smpl_dict


def match_w_hungarian(joints_3d_rel, joints_3d_sla_rel, clip=True):
    distances = []
    for j3d_rel in joints_3d_rel:
        l2_dist = np.linalg.norm(joints_3d_sla_rel - j3d_rel.numpy(), axis=-1).mean(1)
        distances.append(l2_dist)
    cost_matrix = np.stack(distances, 0)

    # Hungarian algo
    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=False)
    # todo use this clipping?
    if clip:
        col_ind = np.clip(col_ind, 0, len(joints_3d_rel) - 1)
    return row_ind, col_ind


def main(args, debug=False):
    SLA_ROOT = "/sailhome/nugrinov/code/CVPR_2024/slahmr_release/slahmr/videos"
    VIDEOS_ROOT = "/sailhome/nugrinov/code/CVPR_2024/slahmr_release/slahmr/videos"

    data_res = [{}, {}]
    count = 0
    limit = args.limit
    out_path = './sample_data/chi3d/'

    # filter for debugging
    # SEQS = [c for c in SEQS if 'Hug' in c]
    # person_name = f"_p{args.person_id}"
    # dataset_path = f'./data/chi3d/{args.fname}{person_name}.pkl'
    # dataset_data = read_pickle(dataset_path)
    dataset_path_1 = f'./data/chi3d/{args.fname}_p1.pkl'
    dataset_path_2 = f'./data/chi3d/{args.fname}_p2.pkl'
    dataset_data_p1 = read_pickle(dataset_path_1)
    dataset_data_p2 = read_pickle(dataset_path_2)
    dataset_data_all = [dataset_data_p1, dataset_data_p2]

    sub_name = args.fname.split('_')[1]
    SEQS = list(SEQ_LENGTHS[sub_name])
    pbar = tqdm(SEQS)
    for idx, seq in enumerate(pbar):
        seq_name = seq[4:]

        if args.filter_seq is not None:
            if args.filter_seq not in seq_name:
                continue

        pbar.set_description(seq)
        if limit == -1:
            pass
        elif idx > limit:
            print(f"*** Warning: Breaking after {limit} seq ***")
            continue
        if not os.path.exists(dataset_path_1) or not os.path.exists(dataset_path_2):
            print("Empty motion files for seq:", seq, idx)
            continue

        # do this for each person
        smpl_dict = []
        for n, dataset_data in enumerate(dataset_data_all):
            # data --> keys: (['s02_Grab_1', 's02_Grab_10', 's02_Grab_11'])
            # motion --> keys: ['trans', 'root_orient', 'pose_body', 'betas', 'joints2d', 'cam2world']
            motion = dataset_data[seq]
            camera_pose = motion["cam2world"]
            seq_length = SEQ_LENGTHS[sub_name][seq]
            # CHI3D has no gender data
            smpl_dict_this = get_smpl_empty_dict(seq_length)

            # motion --> contains ['betas', 'trans', 'root_orient', 'pose_body', 'floor_plane', 'contacts']
            # from motion data, emb uses --> root_orient, pose_body, trans, betas
            # NOTE: pose is 72 dim, but root_orient+pose_body is 66 dim, so the last 6 dims are not used and set to 0
            smpl_dict_this['pose'][:, :3] = motion['root_orient']
            smpl_dict_this['pose'][:, 3:66] = motion['pose_body']
            smpl_dict_this['trans'][:] = motion['trans']
            smpl_dict_this['shape'][:][:, :10] = motion['betas']  # Betas!!
            smpl_dict_this['joints2d'] = motion["joints2d"] # these joints are actually not used!
            smpl_dict.append(smpl_dict_this)


        # read PHALP data
        # use PHALP bbox here, so that it can work with multiple people
        data_name = 'chi3d'
        subject = sub_name
        phalp_data = read_PHALP_data(VIDEOS_ROOT, data_name, subject, seq_name)

        #### Taking smpl init from SLAHMR
        slahmr_dict = read_SLAHMR_data(data_name, subject, seq_name, rot_180z=True)
        if slahmr_dict is None:
            tqdm.write(f"skipping - {subject} - {seq_name}")
            continue
        rotation = slahmr_dict['rotation']
        translation = slahmr_dict['translation']
        focal_length = slahmr_dict['focal_length']
        camera_center = slahmr_dict['camera_center']
        res_dict = slahmr_dict['res_dict']
        pose_aa = slahmr_dict['pose_aa']
        trans_orig = slahmr_dict['trans_orig']
        betas = slahmr_dict['betas']

        ### Assemble camera
        cam = assemble_camera(focal_length, camera_center, rotation, translation, dyncam=args.dyncam)

        # parse SLAHMR smpl data, get joints and project joints, this is multi-people
        pose_aa, trans = slahmr_to_world(pose_aa, trans_orig, betas, res_dict)

        # these are from SLAHMR, parse data and take the meshes to the floor level
        parsed_smpl_dict = prepare_multiperson_smpl_data(pose_aa, trans, betas, cam, to_floor=args.to_floor)
        pose_aa = parsed_smpl_dict['pose_aa']
        betas = parsed_smpl_dict['betas']
        trans = parsed_smpl_dict['trans']
        jts_img_all = parsed_smpl_dict['jts_img_all']
        joints_3d_sla = parsed_smpl_dict['joints_3d']

        # do the matching first with SLAHMR joints and PHALP joints
        kpts_this_person = get_phalp_matched_kpts_simple_all(phalp_data, seq_name, jts_img_all, data_name).squeeze()
        # then do matching between GT smpl and vitpose 2dkpts (and SLAHMR smpl)
        joints_3d_sla_rel = joints_3d_sla[:, 0, 0, :25]
        joints_3d_sla_rel = joints_3d_sla_rel - joints_3d_sla_rel[:, 0, None]

        # merge the two smpl dicts
        smpl_dict_stc = {k: np.stack([dic[k] for dic in smpl_dict]) for k in smpl_dict[0]}
        # p0 = smpl_dict_['pose'][0]
        # p1 = smpl_dict_['pose'][1]

        #get joints-3d for GT SMPL
        (verts_gt, joints_3d_gt), faces = smpl_to_verts(smpl_dict_stc['pose'][:, 0], smpl_dict_stc['trans'][:, 0],
                                          betas=smpl_dict_stc['shape'][:, 0, :10], return_joints=True)
        joints_3d_rel = joints_3d_gt[0, :, :25]
        joints_3d_rel = joints_3d_rel - joints_3d_rel[:, 0, None] #.shape

        # l2_dist = np.linalg.norm(joints_3d_sla_rel - joints_3d_rel.numpy(), axis=-1).mean(1)
        # min_id = np.argmin(l2_dist)
        # match SMPL GT with SLAHMR SMPL and thus with PHALP vitpose 2dkpts
        row_ind, col_ind = match_w_hungarian(joints_3d_rel, joints_3d_sla_rel)
        kpts_ord = kpts_this_person[:, row_ind]
        # k1 = kpts_ord[0, 0]
        # k2 = kpts_ord[0, 1]
        kpts_assigned = kpts_ord[:, col_ind]
        # k1a = kpts_assigned[0, 0]
        # k2a = kpts_assigned[0, 1]

        # try:
        #     kpts_this_person = kpts_this_person[:, min_id]
        # except:
        #     # todo use this clipping? change later
        #     min_id = np.clip(min_id, 0, len(kpts_this_person[0]) - 1)
        #     kpts_this_person = kpts_this_person[:, min_id]
        #     print(f"*** Warning: Clipping min_id to {min_id}, probably SLAHMR produced more than 2 SMPL bodies ***")

        for n, min_id in enumerate(col_ind):
            # todo check why it is not init with SLAHMR poses
            # kpts_this_person = kpts_assigned[:, min_id] # this does NOT give correct 2dkpts order
            kpts_this_person = kpts_assigned[:, n] # this gives correct 2dkpts order
            # overwrite first pose with SLAHMR, this will be used for the first frame initialization
            gt_len = smpl_dict[n]['pose'].shape[0]
            if args.init_slahmr:
                nfra = 5
                smpl_dict[n]['pose'][:nfra] = pose_aa[min_id, :nfra]
                smpl_dict[n]['trans'][:nfra] = trans[min_id, :nfra]
                smpl_dict[n]['shape'][:nfra, :10] = betas[min_id, :nfra]

            jts_2d = kpts_this_person[:gt_len]
            if len(jts_2d.shape)!=3:
                print(f"jts_2d.shape: {jts_2d.shape} for seq {seq_name}: skipping...")
                continue
            smpl_dict[n]['joints2d'] = kpts_this_person[:gt_len]  # (219, 25, 3)

            if debug:
                from utils.misc import save_pointcloud, plot_joints_cv2
                inspect_path = f"inspect_out/process/chi3d/joints3d"
                save_pointcloud(joints_3d_gt[0, 0, :25], Path(f"{inspect_path}/joints_3d{n}.ply"))
                save_pointcloud(joints_3d_sla[min_id, :25], Path(f"{inspect_path}/joints_3d_sla{n}.ply"))

                im_path = f"/sailhome/nugrinov/code/CVPR_2024/slahmr_release/slahmr/videos/{data_name}/train/s03/images/{seq_name}/000001.jpg"
                img = read_image_PIL(im_path)
                plot_joints_cv2(img, kpts_this_person[0, None], show=True, with_text=True)
                plot_joints_cv2(img, jts_img_all[0, None], show=True, with_text=True)

                save_trimesh(verts_gt[0, n], faces, f"inspect_out/process_chi3d/{seq_name}/gt_{n}.ply")
                verts_sla, faces = smpl_to_verts(smpl_dict[n]['pose'], smpl_dict[n]['trans'], smpl_dict[n]['shape'][:, :10])

                save_trimesh(verts_sla[0, 0], faces, f"inspect_out/process_chi3d/{seq_name}/sla_matched_{n}.ply")


            entry = smpl_2_entry(seq, smpl_dict[n], camera_pose)

            if args.init_slahmr:
                # overwrite camera with SLAHMR camera
                entry['cam'].update(cam[0])

            if args.debug:
                count += 1
                excluded = ["betas", "cam", "points3d", "gender", "seq_name"]
                for k, v in entry.items():
                    if k in excluded:
                        continue
                    entry[k] = v[:31]
                if count > 2*limit:
                    break

            data_res[n][seq] = entry

    for n in range(2):
        print(data_res[n].keys())
        save_file(args, out_path, data_res[n], n+1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=["prox", "proxd", "video"], default="proxd")
    parser.add_argument('--fname', type=str)

    parser.add_argument('--sub_name', type=str)
    parser.add_argument('--debug', type=int, choices=[0, 1], default=0)
    parser.add_argument('--to_floor', type=int, choices=[0, 1], default=0)
    parser.add_argument('--smooth_2d', type=int, choices=[0, 1], default=0)
    parser.add_argument('--mask_2d', type=int, choices=[0, 1], default=0)
    parser.add_argument('--person_id', type=int, choices=[1, 2], default=1)
    parser.add_argument('--joints_source', type=str, choices=["dekr", "smpl_gt", "phalp"], default="phalp")
    parser.add_argument('--limit', type=int, default=-1)
    parser.add_argument('--subset', type=int, choices=[0, 1], default=0)
    parser.add_argument('--dyncam', type=int, choices=[0, 1], default=0)
    parser.add_argument('--init_slahmr', type=int, choices=[0, 1], default=0)
    parser.add_argument('--filter', type=str, default=None)
    args = parser.parse_args()
    
    main(args)

    
