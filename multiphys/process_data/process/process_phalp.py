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
from utils.misc import save_pointcloud
from multiphys.process_data.preprocess.dynamic_cam_projection import get_slahmr_emb2cam_transform
from utils.net_utils import replace_slahmr_path
from multiphys.process_data.preprocess.smpl_transforms import rot_pose
from multiphys.process_data.preprocess.smpl_transforms import rot_and_correct_smplx_offset
from tqdm import tqdm
from multiphys.process_data.process.process import get_files_from_dir, get_pose_aa_from_slahmr_init

np.set_printoptions(precision=4)

smplx_folder = '/home/nugrinovic/code/CVPR_2024/EmbodiedPose/data'


def main(args):
    VIDEOS_ROOT = "/sailhome/nugrinov/code/CVPR_2024/slahmr_release/slahmr/videos"
    int_actions = ["Hug", "Push", "Posing"]
    count = 0
    if args.debug:
        print("*** DEBUG MODE ***")
    new_dict = {}
    data_name = args.data_name
    seq_name = args.vid_name
    subject = args.subject
    # get list of video files
    if seq_name is None:
        video_files = get_files_from_dir(VIDEOS_ROOT, data_name, subject)
    else:
        video_files = [Path(f"{VIDEOS_ROOT}/{data_name}/videos/{seq_name}.mp4")]

    for vfile in tqdm(video_files):
        # if "011520_mpii_train" not in str(vfile):
        #     continue
        seq_name = vfile.stem
        seq_name = seq_name.replace(" ", "_")

        act = seq_name.split("_")[0]
        if not act in int_actions and data_name == 'chi3d':
            tqdm.write(f"skipping - {subject} - {seq_name}")
            continue

        # use PHALP bbox here, so that it can work with multiple people
        if data_name == 'chi3d':
            phalp_path = f"{VIDEOS_ROOT}/{data_name}/train/{subject}/slahmr/phalp_out/results/{seq_name}.pkl"
        else:
            phalp_path = f"/home/nugrinovic/code/NEURIPS_2023/slahmr/videos/{data_name}/slahmr/phalp_out/results/{seq_name}.pkl"
            phalp_path = replace_slahmr_path(phalp_path)
        try:
            data = joblib.load(phalp_path)
        except:
            tqdm.write(f"skipping {seq_name}, phalp results missing. Generate in slahmr!")
            continue
        tqdm.write(f"read {seq_name} PHALP data ...")
        B = len(data)
        person_id = args.person_id - 1

        #### Taking smpl init from SLAHMR
        try:
            rotation, translation, focal_length, camera_center, res_dict = get_slahmr_emb2cam_transform(data_name, seq_name, False, subject)
        except:
            tqdm.write(f"skipping {seq_name}, scene_dict data does not exist! generate in slahrm")
            continue
        try:
            pose_aa, trans_orig, betas = get_pose_aa_from_slahmr_init(res_dict, person_id, all_frames=True)
        except:
            tqdm.write(f"skipping {seq_name}, person_id {person_id} does not exist! due to slahrm estimate (discard for now)")
            continue

        Bpose = pose_aa.shape[0]
        B = min(B, Bpose)

        if 0:
            verts, faces = smpl_to_verts(pose_aa, trans_orig, betas=None)
            out_path = f"inspect_out/phalp_process/{seq_name}/original_{person_id}.ply"
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            save_trimesh(verts[0, 0], faces, out_path)

        betas = torch.Tensor(betas).float()  # (1, 10)
        ## Apply the rotation to make z the up direction
        final_Rt = res_dict['final_Rt'].cpu()  # [0][None]
        pose_aa_i = torch.tensor(pose_aa)
        trans_i = torch.tensor(trans_orig)
        # transform to world
        pose_aa_w_i = rot_pose(pose_aa_i.cpu(), final_Rt.cpu())
        trans_w_i, verts_i_w, faces = rot_and_correct_smplx_offset(pose_aa_w_i, trans_i, betas, final_Rt, get_verts=True)
        pose_aa = pose_aa_w_i.clone()
        trans = trans_w_i.clone()

        if 0:
            # save smpl mesh#
            verts, faces = smpl_to_verts(pose_aa, trans, betas=betas[:, :10])
            save_trimesh(verts[0, 0], faces, f"inspect_out/phalp_process/{seq_name}/original_rot_{person_id}.ply")

        # take the mesh to the floor; we then need to update camera params
        trans = trans_body_to_floor(pose_aa, trans, betas)
        # def placeholders
        pose_aa_ph = pose_aa.clone()
        trans_ph = trans.copy()
        kp_25 = np.zeros([B, 25, 3])

        if 0:
            # save smpl mesh
            verts, faces = smpl_to_verts(pose_aa, trans, betas=betas[:, :10])
            out_path = f"inspect_out/phalp_process/{seq_name}/phalp_{person_id}.ply"
            save_trimesh(verts[0, 0], faces, out_path)

        ### Assemblle camera
        cam = assemble_camera(focal_length, camera_center, rotation, translation, dyncam=args.dyncam)
        pose_mat = sRot.from_rotvec(pose_aa_ph.reshape(B * 24, 3)).as_matrix().reshape(B, 24, 3, 3)
        pose_body = pose_mat[:, 1:22]
        root_orient = pose_mat[:, 0:1]
        trans = trans_ph.copy()

        start = 0
        end = B
        # PHALP estimates, nned to do matching between phalp joints and the intial SLAHMR smpl bodies
        (_, joints), _ = smpl_to_verts(pose_aa[0, None], trans[0, None], betas=betas[:, :10], return_joints=True)
        joints = joints[0, 0, :25].detach().numpy()
        # get smpl joints in image space
        jts_img = project_jts(cam[0], joints)
        # op_kpts = get_phalp_kpts(data, seq_name)
        kpts_this_person = get_phalp_kpts_simple(data, seq_name, jts_img, data_name).squeeze()
        # kpts_this_person = op_kpts[:, person_id] # (76, 25, 3)
        # Overwrite the joints with the ones from PHALP
        kp_25[start:end] = kpts_this_person[:end]

        if 0:
            out = smpl_to_verts(pose_aa, trans, betas=betas[:, :10], return_joints=True)
            (verts, joints), faces = out
            joints = joints[0, 0, :24].detach().numpy()
            # this is the same as phalp.ply so it is ok!
            out_path = f"inspect_out/phalp_process/phalp_proj_{person_id}.ply"
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            save_trimesh(verts[0, 0], faces, out_path)
            # joints are also correct
            full_R_sla = cam["full_R"]
            full_t_sla = cam["full_t"]
            K_sla = cam["K"]
            out_path = f"inspect_out/phalp_process/joints_{person_id}.ply"
            save_pointcloud(joints, out_path)
            jts_cam = (full_R_sla @ joints.T + full_t_sla[:, None]).T
            # save_pointcloud(jts_cam, "inspect_out/phalp_process/jts_cam.ply")
            jts_img = (K_sla @ jts_cam.T).T
            jts_img = jts_img[:, :2] / jts_img[:, 2:]
            jts_img = jts_img.astype(np.int32)
            im_path = f"/home/nugrinovic/code/NEURIPS_2023/slahmr/videos/viw/images/{seq_name}/000001.jpg"
            im_path = replace_slahmr_path(im_path)
            img = read_image_PIL(im_path)
            plot_joints_cv2(img, jts_img[None], show=True)

        # vitpose[0, 19]
        if 0:
            from utils.misc import plot_joints_cv2
            from utils.misc import read_image_PIL
            idx = 0
            im_path = f"/home/nugrinovic/code/NEURIPS_2023/slahmr/videos/viw/images/{seq_name}"
            im_path = replace_slahmr_path(im_path)
            res_img_path = Path(im_path)
            img_files = sorted(res_img_path.glob("*.jpg"))
            img = read_image_PIL(img_files[idx])
            plot_joints_cv2(img, kpts_this_person[idx, None], show=True, with_text=True, sc=2)

        key_name = f"{subject}_{seq_name}" if data_name == 'chi3d' else seq_name

        new_dict[key_name] = {
            "joints2d": kp_25[start:end].copy(), # (B, 25, 3)
            "pose_body": pose_body[start:end],  # (B, 21, 3, 3)
            "root_orient": root_orient[start:end].squeeze(), # (B, 3, 3)
            "trans": trans.squeeze()[start:end], # (B, 3)
            "pose_aa" : pose_aa_ph.reshape(-1, 72)[start:end], # (B, 72)
            "joints": np.zeros([B, 22, 3]),  # (B, 66)
            "pose_6d": np.zeros([B, 24, 6]),
            'betas': betas.numpy().squeeze(), # (16, )
            "gender": "neutral",
            "seq_name": key_name,
            "trans_vel": np.zeros([B, 1, 3]), # (B, 3)
            "joints_vel": np.zeros([B, 22, 3]),
            "root_orient_vel": np.zeros([B, 1, 3]),
            "points3d": None,
            "cam": cam
        }

        new_dict[key_name]["video_path"] = str(vfile)
        if args.debug:
            count += 1
            excluded = ["betas", "cam", "points3d", "gender", "seq_name"]
            for k, v in new_dict[key_name].items():
                if k in excluded:
                    continue
                new_dict[key_name][k] = v[:31]

            if count > 1:
                break

        # new_dict[seq_name]["video_path"] = args.video_path + f"/{seq_name}.mp4"

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    subj_name = f"_{subject}" if data_name =='chi3d' else ""
    data_n = data_name + "_slahmr" if data_name == 'chi3d' else data_name
    debug_name = "_debug" if args.debug else ""
    person_id_name = f"_p{args.person_id}"
    bbox_name = f"_{args.bbox_source}Box"
    dyncam_name = "_dyncam" if args.dyncam else ""
    out_fpath = Path(args.output_dir) / f"{data_n}{subj_name}{debug_name}{person_id_name}{bbox_name}{dyncam_name}_slaInit_slaCam.pkl"
    joblib.dump(new_dict, out_fpath)
    print(f"Saved at : {out_fpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default='viw', choices=['viw', 'chi3d', 'hi4d', 'shorts'])
    # parser.add_argument("--video_path", type=str)
    parser.add_argument("--vid_name", type=str, default=None)
    parser.add_argument("--subject", type=str, default=None)
    parser.add_argument("--output_dir", default='sample_data/videos_wild')
    parser.add_argument('--debug', type=int, choices=[0, 1], default=0)
    parser.add_argument("--joints_source", type=str, choices=['hybrik', 'phalp'], default='phalp')
    parser.add_argument("--bbox_source", type=str, choices=['hybrik', 'phalp'], default='phalp')
    parser.add_argument('--person_id', type=int, choices=[1, 2])
    parser.add_argument('--dyncam', type=int, choices=[0, 1], default=0)
    args = parser.parse_args()

    main(args)