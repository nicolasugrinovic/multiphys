import argparse
import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from scipy.spatial.transform import Rotation as sRot
import joblib
# from embodiedpose.models.humor.body_model.utils import smpl_to_openpose
# from embodiedpose.models.humor.utils.humor_mujoco import SMPL_2_OP, OP_14_to_OP_12
# from uhc.smpllib.smpl_parser import SMPL_Parser, SMPLH_BONE_ORDER_NAMES, SMPLH_Parser
from pathlib import Path
# from utils.misc import read_json
from utils.smpl import smpl_to_verts
from utils.misc import save_trimesh
from utils.misc import save_pointcloud
# from pyquaternion import Quaternion as Q
# import trimesh
# from utils.misc import read_npy
from utils.misc import plot_joints_cv2
from utils.misc import read_image_PIL

from utils.net_utils import replace_slahmr_path
# from utils.torch_utils import to_numpy
from tqdm import tqdm

from multiphys.process_data.process.process import get_files_from_dir
from multiphys.process_data.process.process import assemble_camera
from multiphys.process_data.process.process import get_phalp_matched_kpts_simple_all

from multiphys.process_data.process.process import prepare_multiperson_smpl_data
from multiphys.process_data.process.process import slahmr_to_world
from multiphys.process_data.process.process import read_SLAHMR_data
from multiphys.process_data.process.process import read_PHALP_data

np.set_printoptions(precision=4)
smplx_folder = '/home/nugrinovic/code/CVPR_2024/EmbodiedPose/data'


def main(args, debug=False):
    VIDEOS_ROOT = "/sailhome/nugrinov/code/CVPR_2024/slahmr_release/slahmr/videos"
    int_actions = ["Hug", "Push", "Posing"]
    count = 0
    if args.debug:
        print("*** DEBUG MODE ***")
    new_dict = [{}, {}]
    data_name = args.data_name
    seq_name = args.vid_name
    subject = args.subject
    filter = args.filter_seq
    # get list of video files
    if seq_name is None:
        video_files = get_files_from_dir(VIDEOS_ROOT, data_name, subject)
    else:
        video_files = [Path(f"{VIDEOS_ROOT}/{data_name}/videos/{seq_name}.mp4")]
    print(f"Found {len(video_files)} videos for {data_name} {subject}")

    for vfile in tqdm(video_files):
        # if "pair15_1_backhug15" not in str(vfile):
        #     continue
        seq_name = vfile.stem
        seq_name = seq_name.replace(" ", "_")
        act = seq_name.split("_")[0]

        if filter is not None:
            if filter not in seq_name:
                continue

        if not act in int_actions and data_name == 'chi3d' and args.filter_actions:
            tqdm.write(f"skipping - {subject} - {seq_name}")
            continue

        # use PHALP bbox here, so that it can work with multiple people
        phalp_data = read_PHALP_data(VIDEOS_ROOT, data_name, subject, seq_name)
        if phalp_data is None:
            tqdm.write(f"NO phalp_light, skipping - {subject} - {seq_name}")
            continue
        tqdm.write(f"read {seq_name} PHALP data ...")
        B = len(phalp_data)

        #### Taking smpl init from SLAHMR
        slahmr_dict = read_SLAHMR_data(data_name, subject, seq_name)
        if slahmr_dict is None:
            tqdm.write(f"skipping - {subject} - {seq_name}")
            continue

        res_dict = slahmr_dict['res_dict']
        pose_aa = slahmr_dict['pose_aa']
        trans_orig = slahmr_dict['trans_orig']
        betas = slahmr_dict['betas']

        Bpose = pose_aa.shape[1]
        B = min(B, Bpose)
        # if 0:
        #     verts, faces = smpl_to_verts(pose_aa, trans_orig, betas=None)
        #     out_path = f"inspect_out/phalp_process/{seq_name}/original_{person_id}.ply"
        #     Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        #     save_trimesh(verts[0, 0], faces, out_path)
        
        pose_aa, trans = slahmr_to_world(pose_aa, trans_orig, betas, res_dict)

        ### Assemblle camera
        cam = assemble_camera(slahmr_dict, dyncam=args.dyncam)

        # parse SLAHMR smpl data, get joints and project joints
        parsed_smpl_dict = prepare_multiperson_smpl_data(pose_aa, trans, betas, cam)
        pose_aa = parsed_smpl_dict['pose_aa']
        betas = parsed_smpl_dict['betas']
        trans = parsed_smpl_dict['trans']
        jts_img_all = parsed_smpl_dict['jts_img_all']

        kpts_this_person = get_phalp_matched_kpts_simple_all(phalp_data, seq_name, jts_img_all, data_name).squeeze()

        # def placeholders
        kp_25 = np.zeros([B, 25, 3])
        start, end = 0, B
        # Format data for saving
        for n in range(len(pose_aa[:2])):
            try:
                kp_25[start:end] = kpts_this_person[:end, n]
            except:
                print("Most likely, SLAHMR produced only one person in the first frame1, skipping ...")
                continue
            pose_mat = sRot.from_rotvec(pose_aa[n].reshape(B * 24, 3)).as_matrix().reshape(B, 24, 3, 3)
            pose_body = pose_mat[:, 1:22]
            root_orient = pose_mat[:, 0:1]

            if debug:
                out = smpl_to_verts(pose_aa[n], trans[n], betas=betas[n][:, :10], return_joints=True)
                (verts, joints), faces = out
                joints = joints[0, 0, :24].detach().numpy()
                # this is the same as phalp.ply so it is ok!
                out_path = f"inspect_out/phalp_process/sla_init_{n}.ply"
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


            key_name = f"{subject}_{seq_name}" if data_name == 'chi3d' or data_name == 'expi' else seq_name
            new_dict[n][key_name] = {
                "pose_aa": pose_aa[n].reshape(-1, 72)[start:end],  # (B, 72)
                "pose_6d": np.zeros([B, 24, 6]),
                "pose_body": pose_body[start:end],  # (B, 21, 3, 3)
                "trans": trans[n].squeeze()[start:end],  # (B, 3)
                "trans_vel": np.zeros([B, 1, 3]),  # (B, 3)
                "root_orient": root_orient[start:end].squeeze(),  # (B, 3, 3)
                "root_orient_vel": np.zeros([B, 1, 3]),
                "joints": np.zeros([B, 22, 3]),  # (B, 66)
                "joints_vel": np.zeros([B, 22, 3]),
                'betas': betas[n].squeeze(), # (16, )
                "seq_name": key_name,
                "gender": "neutral",
                "joints2d": kp_25[start:end].copy(),  # (B, 25, 3)
                "points3d": None,
                "cam": cam
            }

        count += 1
        for n in range(len(pose_aa[:2])):
            new_dict[n][key_name]["video_path"] = str(vfile)
            if args.debug:
                excluded = ["betas", "cam", "points3d", "gender", "seq_name"]
                for k, v in new_dict[n][key_name].items():
                    if k in excluded:
                        continue
                    new_dict[n][key_name][k] = v[:31]

        if args.debug and count > 1:
            break

    # if len(pose_aa[:2])<2:
    #     print("*** WARNING: I have pose for only 1 person, only p1 file will be saved ***")

    print(f"processed {count} sequences")
    # for n in range(len(pose_aa[:2])):
    for n in range(2):
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        subj_name = f"_{subject}" if data_name =='chi3d' or data_name=='expi' else ""
        data_n = data_name + "_slahmr" if data_name == 'chi3d' else data_name
        debug_name = "_debug" if args.debug else ""
        person_id_name = f"_p{n+1}"
        bbox_name = f"_{args.bbox_source}Box"
        dyncam_name = "_dyncam" if args.dyncam else ""
        filt_n = f"_{args.filter_seq}" if filter else "_all"
        out_fpath = Path(args.output_dir) / f"{data_n}{subj_name}{debug_name}{person_id_name}{bbox_name}{dyncam_name}{filt_n}_slaInit_slaCam.pkl"
        joblib.dump(new_dict[n], out_fpath)
        print(new_dict[n].keys())
        print(f"Saved at : {out_fpath}")
        # pose1 = new_dict[0]['_pair00_1_dance00']["pose_body"]
        # pose2 = new_dict[1]['_pair00_1_dance00']["pose_body"]

    print("Done")
    if 0:
        out = smpl_to_verts(pose_aa[n], trans[n], betas=betas[n][:, :10], return_joints=True)
        (verts, joints), faces = out
        joints = joints[0, 0, :24].detach().numpy()
        # this is the same as phalp.ply so it is ok!
        out_path = f"inspect_out/phalp_process/{data_name}/sla_init_{n}.ply"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        save_trimesh(verts[0, 0], faces, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default='viw', choices=['viw', 'chi3d', 'hi4d', 'expi', 'shorts'])
    parser.add_argument("--vid_name", type=str, default=None)
    parser.add_argument("--subject", type=str, default='.')
    parser.add_argument("--output_dir", default='sample_data/videos_wild')
    parser.add_argument('--debug', type=int, choices=[0, 1], default=0)
    parser.add_argument("--joints_source", type=str, choices=['hybrik', 'phalp'], default='phalp')
    parser.add_argument("--bbox_source", type=str, choices=['hybrik', 'phalp'], default='phalp')
    # parser.add_argument('--person_id', type=int, choices=[1, 2])
    parser.add_argument('--dyncam', type=int, choices=[0, 1], default=0)
    parser.add_argument("--filter", type=str, default=None)
    parser.add_argument("--filter_actions", type=int, choices=[0, 1], default=0)
    args = parser.parse_args()

    main(args)