import argparse
import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from scipy.spatial.transform import Rotation as sRot
import joblib
from embodiedpose.models.humor.body_model.utils import smpl_to_openpose
# from embodiedpose.models.humor.utils.humor_mujoco import SMPL_2_OP, OP_14_to_OP_12
# from uhc.smpllib.smpl_parser import SMPL_Parser, SMPLH_BONE_ORDER_NAMES, SMPLH_Parser
import torch
from pathlib import Path
from utils.smpl import smpl_to_verts
from utils.misc import save_trimesh
from utils.misc import save_pointcloud
# from pyquaternion import Quaternion as Q
import trimesh
# from utils.misc import read_npy
from utils.misc import plot_joints_cv2
from utils.misc import read_image_PIL

from multiphys.process_data.preprocess.dynamic_cam_projection import get_slahmr_emb2cam_transform
from utils.net_utils import replace_slahmr_path
from utils.torch_utils import to_numpy
from multiphys.process_data.preprocess.smpl_transforms import rot_pose
from multiphys.process_data.preprocess.smpl_transforms import rot_and_correct_smplx_offset

np.set_printoptions(precision=4)

smplx_folder = '/home/nugrinovic/code/CVPR_2024/EmbodiedPose/data'

def assemble_camera(focal_length, camera_center, rotation, translation):
    ### Assemblle camera
    focal_length = focal_length.cpu()
    camera_center = camera_center.cpu()
    # for now one cam, todo change per-frame
    rotation = rotation[0]
    translation = translation[0]
    full_R_sla = rotation
    full_t_sla = translation #  + np.array([0, -diff_trans, 0])
    K_sla = np.array([[focal_length[0, 0], 0, camera_center[0, 0]],
                      [0, focal_length[0, 1], camera_center[0, 1]],
                      [0., 0., 1.]])
    cam = {
        "full_R": full_R_sla,
        "full_t": full_t_sla,
        "K": K_sla,
        'img_w': 2 * camera_center[0, 0],
        'img_h': 2 * camera_center[0, 1],
        'scene_name': None
    }
    cam = to_numpy(cam)
    return cam

def trans_body_to_floor(pose_aa, trans, betas):
    # take the mesh to the floor; we then need to update camera params
    verts, faces = smpl_to_verts(pose_aa, trans, betas=betas[:, :10])
    mesh = trimesh.Trimesh(vertices=verts[0, 0].detach().cpu().numpy(), faces=faces)
    bbox = mesh.bounding_box.bounds
    # while this works here, this is not totally correct as it ignores the smplx offset and thus
    # hurts later when the camera is used. So we need to correct smplx offset above
    diff_trans = bbox[0][None, 2]
    trans_floor_z = trans[:, 2] - diff_trans
    new_trans = np.concatenate([trans[:, :2], trans_floor_z[:, None]], 1)
    trans = new_trans.copy()
    return trans


def get_pose_aa_from_slahmr_init(res_dict, person_id, all_frames=False):
    # originally getting only the 1st frame info, but to override we need all
    if all_frames:
        T = res_dict["pose_body"].shape[1]
        body_pose_ = res_dict["pose_body"][person_id].cpu().numpy()
        betas = res_dict["betas"][person_id, None, :10].cpu().numpy()
        trans_orig = res_dict["trans"][person_id].cpu().numpy()
        global_orient = res_dict["root_orient"][person_id].cpu().numpy()
        body_pose = np.zeros([T, 69]).astype(np.float32)
        body_pose[:, :63] = body_pose_
        pose_aa = np.concatenate([global_orient, body_pose], axis=1)
    else:
        body_pose_ = res_dict["pose_body"][person_id, 0].cpu().numpy()
        betas = res_dict["betas"][person_id, None, :10].cpu().numpy()
        trans_orig = res_dict["trans"][person_id, 0, None].cpu().numpy()
        global_orient = res_dict["root_orient"][person_id, 0, None].cpu().numpy()
        body_pose = np.zeros([1, 69]).astype(np.float32)
        body_pose[:, :63] = body_pose_
        pose_aa = np.concatenate([global_orient, body_pose], axis=1)
    return pose_aa, trans_orig, betas


def get_phalp_kpts(data):
    img_names = sorted(data.keys())
    all_joints = []
    is_complete = []
    for im_name in img_names:
        phalp_data = data[im_name]
        vitpose = phalp_data['vitpose']  # list of 2 with shape (25, 3)
        if 0:
            j2d = np.array(vitpose[0])[:, :2]
            im_path = f"/home/nugrinovic/code/NEURIPS_2023/slahmr/videos/viw/images/{seq_name}/{im_name}"
            img = read_image_PIL(im_path)
            plot_joints_cv2(img, j2d[None], show=True)
        # vitpose only has joints from 0 to 18, so 19 jts and does not have 1 (chest)
        # nor 8 (root), it does contain the nose at 0.
        vitpose_arr = np.zeros([2, 25, 3])
        complete = False
        len_vit = len(vitpose)
        if len_vit > 2:
            print("More than 2 people in the frame")
        for ip, pose_list in enumerate(vitpose[:2]):
            vitpose_arr[ip] = vitpose[ip]
            if ip == 1:
                complete = True
        # vitpose_arr = np.array(vitpose)
        is_complete.append(complete)
        all_joints.append(vitpose_arr)
    op_kpts = np.stack(all_joints, 0)

    # am I using the conf values here? YES, confs are taken directly from the vitpose output
    op_kpts[:, :, 1] = (op_kpts[:, :, 2] + op_kpts[:, :, 5]) / 2
    op_kpts[:, :, 8] = (op_kpts[:, :, 9] + op_kpts[:, :, 12]) / 2
    return op_kpts


def mask_joints_w_vis(j2d):
    vis = j2d[0, :, 2] > 0.1
    j2d_masked = j2d[:, vis]
    return j2d_masked


def get_phalp_kpts_simple(data, seq_name, ref_jts):
    smpl2op_map = smpl_to_openpose("smpl",
                                    use_hands=False,
                                    use_face=False,
                                    use_face_contour=False,
                                    openpose_format='coco25')

    img_names = sorted(data.keys())
    all_joints = []
    for im_name in img_names:
        phalp_data = data[im_name]
        vitpose = phalp_data['vitpose']  # list of 2 with shape (25, 3)

        if 0:
            j2d = np.array(vitpose[0])[:, :2]
            im_path = f"/home/nugrinovic/code/NEURIPS_2023/slahmr/videos/viw/images/{seq_name}/000001.jpg"
            im_path = replace_slahmr_path(im_path)
            img = read_image_PIL(im_path)
            plot_joints_cv2(img, j2d[None], show=True, with_text=True)
            plot_joints_cv2(img, ref_jts[None], show=True, with_text=True)

        # vitpose only has joints from 0 to 18, so 19 jts and does not have 1 (chest)
        # nor 8 (root), it does contain the nose at 0.
        # choose
        vitpose = np.stack(vitpose) # this is in OP format
        # vitpose_m = mask_joints_w_vis(vitpose)
        # bbox_vit = bbox_from_joints_several(vitpose_m[:, :, :2])
        # areas = compute_area_several(bbox_vit)

        # ref_jts are in SMPL format
        smpl_2op_submap = smpl2op_map[smpl2op_map < 25]
        ref_jts_op = ref_jts[smpl_2op_submap]
        vitpose_14 = vitpose[:, :15]

        # make threshold more than 0.4
        confs_mask = vitpose_14[:, :, 2] > 0.4
        diff = confs_mask * np.abs(vitpose_14[:, :, :2] - ref_jts_op[None, :, :2]).mean(-1)
        diff = diff.mean(-1)
        idx_chosen = diff.argmin()

        # plot_joints_cv2(img, ref_jts_op[None], show=True, with_text=True)
        # plot_joints_cv2(img, vitpose_14, show=True, with_text=True)

        # vitpose_m = mask_joints_w_vis(vitpose)
        # bbox_vit = bbox_from_joints_several(vitpose_m[:, :, :2])
        # # plot_boxes_cv2(img, bbox_vit, do_return=False)
        # bbox_ref = bbox_from_joints(ref_jts)
        # # plot_boxes_cv2(img, bbox_ref[None], do_return=False)
        # ious = box_iou_np(bbox_ref[None], bbox_vit)

        # idx_chosen = ious.argmax()
        vitpose_arr = np.zeros([1, 25, 3])
        vitpose_arr[0] = vitpose[idx_chosen]

        # np.set_printoptions(precision=4)
        # vitpose_arr[0, 11]
        # vitpose_arr[0, 14]
        # vitpose_arr[0, 10]
        # taking care of confs
        confs = vitpose_arr[0, :, 2]
        confs = [c if c > 0.4 else 0. for c in confs]
        vitpose_arr[0, :, 2] = confs
        all_joints.append(vitpose_arr)

    op_kpts = np.stack(all_joints, 0)
    # am I using the conf values here? YES, confs are taken directly from the vitpose output
    op_kpts[:, :, 1] = (op_kpts[:, :, 2] + op_kpts[:, :, 5]) / 2
    op_kpts[:, :, 8] = (op_kpts[:, :, 9] + op_kpts[:, :, 12]) / 2

    if 0:
        plot_joints_cv2(img, ref_jts_op[None], show=True, with_text=True)
        plot_joints_cv2(img, op_kpts[0], show=True, with_text=True)

    return op_kpts

def project_jts(cam, joints):
    """ this was used for convenience, please use perspective_projection() in any other case"""
    full_R_sla = cam["full_R"]
    full_t_sla = cam["full_t"]
    K_sla = cam["K"]
    # out_path = f"inspect_out/phalp_process/joints_{person_id}.ply"
    # save_pointcloud(joints, out_path)
    jts_cam = (full_R_sla @ joints.T + full_t_sla[:, None]).T
    jts_img = (K_sla @ jts_cam.T).T
    jts_img = jts_img[:, :2] / jts_img[:, 2:]
    return jts_img

def get_files_from_dir(VIDEOS_ROOT, data_name):
    # VIDEOS_ROOT = "/sailhome/nugrinov/code/CVPR_2024/slahmr_release/slahmr/videos"
    vid_dir = f"{VIDEOS_ROOT}/{data_name}/videos"
    video_files = sorted(Path(vid_dir).glob("*.mp4"))
    return video_files

def main(args):
    VIDEOS_ROOT = "/sailhome/nugrinov/code/CVPR_2024/slahmr_release/slahmr/videos"

    new_dict = {}

    data_name = args.data_name
    seq_name = args.vid_name

    # get list of video files
    if seq_name is None:
        video_files = get_files_from_dir(VIDEOS_ROOT, data_name)
    else:
        video_files = [Path(f"{VIDEOS_ROOT}/{data_name}/videos/{seq_name}.mp4")]

    for vfile in video_files:
        seq_name = vfile.stem
        # data_dir = "data/smpl"
        # smpl_parser_n = SMPL_Parser(model_path=data_dir, gender="neutral")
        # smpl_parser_m = SMPL_Parser(model_path=data_dir, gender="male")
        # smpl_parser_f = SMPL_Parser(model_path=data_dir, gender="female")

        # use PHALP bbox here, so that it can work with multiple people
        phalp_path = f"/home/nugrinovic/code/NEURIPS_2023/slahmr/videos/{data_name}/slahmr/phalp_out/results/{seq_name}.pkl"
        phalp_path = replace_slahmr_path(phalp_path)
        try:
            print(f"reading {seq_name} PHALP data ...")
            data = joblib.load(phalp_path)
        except:
            print(f"skipping {seq_name}, phalp results missing. Generate in slahmr!")
            continue

        B = len(data)
        person_id = args.person_id - 1
        #### Taking smpl init from SLAHMR
        rotation, translation, focal_length, camera_center, res_dict = get_slahmr_emb2cam_transform(data_name, seq_name)
        pose_aa, trans_orig, betas = get_pose_aa_from_slahmr_init(res_dict, person_id, all_frames=True)

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
            # save smpl mesh#
            verts, faces = smpl_to_verts(pose_aa, trans, betas=betas[:, :10])
            out_path = f"inspect_out/phalp_process/{seq_name}/phalp_{person_id}.ply"
            save_trimesh(verts[0, 0], faces, out_path)

        ### Assemblle camera
        cam = assemble_camera(focal_length, camera_center, rotation, translation)

        pose_mat = sRot.from_rotvec(pose_aa_ph.reshape(B * 24, 3)).as_matrix().reshape(B, 24, 3, 3)
        pose_body = pose_mat[:, 1:22]
        root_orient = pose_mat[:, 0:1]
        trans = trans_ph.copy()


        start = 0
        end = B
        # PHALP estimates, nned to do matching between phalp joints and the intial SLAHMR smpl bodies
        (_, joints), _  = smpl_to_verts(pose_aa[0, None], trans[0, None], betas=betas[:, :10], return_joints=True)
        joints = joints[0, 0, :25].detach().numpy()

        # get smpl joints in image space
        jts_img = project_jts(cam, joints)
        # op_kpts = get_phalp_kpts(data, seq_name)
        kpts_this_person = get_phalp_kpts_simple(data, seq_name, jts_img).squeeze()
        # kpts_this_person = op_kpts[:, person_id] # (76, 25, 3)
        # Overwrite the joints with the ones from PHALP
        kp_25[start:end] = kpts_this_person

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

        new_dict[seq_name] = {
            "joints2d": kp_25[start:end].copy(), # (B, 25, 3)
            "pose_body": pose_body[start:end],  # (B, 21, 3, 3)
            "root_orient": root_orient[start:end].squeeze(), # (B, 3, 3)
            "trans": trans.squeeze()[start:end], # (B, 3)
            "pose_aa" : pose_aa_ph.reshape(-1, 72)[start:end], # (B, 72)
            "joints": np.zeros([B, 22, 3]),  # (B, 66)
            # "seq_name": "01",
            "pose_6d": np.zeros([B, 24, 6]),
            'betas': betas.numpy().squeeze(), # (16, )
            "gender": "neutral",
            "seq_name": seq_name,
            "trans_vel": np.zeros([B, 1, 3]), # (B, 3)
            "joints_vel": np.zeros([B, 22, 3]),
            "root_orient_vel": np.zeros([B, 1, 3]),
            "points3d": None,
            "cam": cam
        }

        if args.debug:
            excluded = ["betas", "cam", "points3d", "gender", "seq_name"]
            for k, v in new_dict[seq_name].items():
                if k in excluded:
                    continue
                new_dict[seq_name][k] = v[:31]

        # new_dict[seq_name]["video_path"] = args.video_path + f"/{seq_name}.mp4"
        new_dict[seq_name]["video_path"] = str(vfile)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    debug_name = "_debug" if args.debug else ""
    person_id_name = f"_p{args.person_id}"
    bbox_name = f"_{args.bbox_source}Box"
    out_fpath = Path(args.output_dir) / f"{seq_name}{debug_name}{person_id_name}{bbox_name}_slaInit_slaCam.pkl"
    joblib.dump(new_dict, out_fpath)
    print(f"Saved at : {out_fpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default='viw', choices=['viw', 'chi3d'])
    # parser.add_argument("--video_path", type=str)
    parser.add_argument("--vid_name", type=str, default=None)
    parser.add_argument("--output_dir", default='sample_data/videos_wild')
    parser.add_argument('--debug', type=int, choices=[0, 1], default=0)
    parser.add_argument("--joints_source", type=str, choices=['hybrik', 'phalp'], default='phalp')
    parser.add_argument("--bbox_source", type=str, choices=['hybrik', 'phalp'], default='phalp')
    parser.add_argument('--person_id', type=int, choices=[1, 2])
    args = parser.parse_args()

    main(args)