import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import joblib
from embodiedpose.models.humor.body_model.utils import smpl_to_openpose
import torch
from pathlib import Path
from utils.smpl import smpl_to_verts
from utils.misc import save_trimesh
import trimesh
from utils.misc import read_image_PIL
from multiphys.data_process.dynamic_cam_projection import get_slahmr_emb2cam_transform
from utils.net_utils import replace_slahmr_path
from utils.torch_utils import to_numpy
from multiphys.data_process.smpl_transforms import rot_pose
from multiphys.data_process.smpl_transforms import rot_and_correct_smplx_offset
from utils.bbox import bbox_from_joints_several
from tqdm import tqdm
from utils.misc import plot_skel_cv2
from utils.misc import plot_boxes_w_persID_cv2
from utils.misc import save_img
from utils.misc import read_pickle
from utils.video import make_video
from scipy.optimize import linear_sum_assignment

np.set_printoptions(precision=4)


def read_PHALP_data(VIDEOS_ROOT, data_name, subject, seq_name):
    # use PHALP bbox here, so that it can work with multiple people
    if data_name == 'chi3d':
        phalp_path = f"{VIDEOS_ROOT}/{data_name}/train/{subject}/phalp/{seq_name}.pkl"
    elif data_name == 'expi':
        phalp_path = f"/home/nugrinovic/code/NEURIPS_2023/slahmr/videos/{data_name}/{subject}/phalp/{seq_name}.pkl"
        phalp_path = replace_slahmr_path(phalp_path)
    else:
        phalp_path = f"/home/nugrinovic/code/NEURIPS_2023/slahmr/videos/{data_name}/phalp/{seq_name}.pkl"
        phalp_path = replace_slahmr_path(phalp_path)
    try:
        phalp_data = joblib.load(phalp_path)
    except:
        tqdm.write(f"skipping {seq_name}, phalp results missing. Generate in slahmr!")
        return None
    return phalp_data


def read_SLAHMR_data(SLA_ROOT, data_name, subject, seq_name, rot_180z=False, sub_dir=None, sla_postfix=None):
    try:
        out = get_slahmr_emb2cam_transform(SLA_ROOT, data_name, seq_name, rot_180z, subject)
        rotation, translation, focal_length, camera_center, res_dict, T_w2c, final_Rt, final_Rt_inv = out
        if np.isnan(rotation).any():
            print(f"skipping {seq_name}, rotation is nan!")
            return None
    except:
        tqdm.write(f"skipping {seq_name}, scene_dict data does not exist! generate in slahrm")
        return None

    if sub_dir is not None and sla_postfix is not None:
        # read scene_dict from post-optim step
        res_dir = res_dict['res_dir']
        res_dir = res_dir.replace('-val', sla_postfix)
        try:
            new_res_dict = read_pickle(f"{res_dir}/{sub_dir}/{seq_name}_scene_dict.pkl")
        except:
            tqdm.write(f"skipping {sub_dir}/{seq_name}, scene_dict data does not exist! generate in slahrm")
            return None
        try:
            # betas here are 0 as they are not optim, read from the "original" slahmr results
            pose_aa, trans_orig, _ = get_pose_aa_from_slahmr_init_all(new_res_dict)
            _, _, betas_new = get_pose_aa_from_slahmr_init_all(res_dict)
            # ALSO betas have to be reversed in order, due to our MultiPhys datalaoder having swap opt as default
            betas = betas_new[[1, 0]]
        except:
            tqdm.write(f"skipping {seq_name}, does not exist! due to slahrm estimate (discard for now)")
            return None
    else:
        try:
            pose_aa, trans_orig, betas = get_pose_aa_from_slahmr_init_all(res_dict)
        except:
            tqdm.write(f"skipping {seq_name}, does not exist! due to slahrm estimate (discard for now)")
            return None

    return_dict = {
        "rotation": rotation,
        "translation": translation,
        "focal_length": focal_length,
        "camera_center": camera_center,
        "res_dict": res_dict,
        "pose_aa": pose_aa,
        "trans_orig": trans_orig,
        "betas": betas,
        "T_w2c": T_w2c,
        "final_Rt": final_Rt,
        "final_Rt_inv": final_Rt_inv,

    }

    return return_dict


def slahmr_to_world(pose_aa, trans_orig, betas, res_dict, debug=False):
    betas = torch.Tensor(betas).float()  # (1, 10)
    ## Apply the rotation to make z the up direction
    final_Rt = res_dict['final_Rt'].cpu()  # [0][None]
    pose_aa_i = torch.tensor(pose_aa)
    trans_i = torch.tensor(trans_orig)
    # transform to world
    pose_aa_w_i = rot_pose(pose_aa_i.cpu(), final_Rt.cpu())
    trans_w_i, verts_i_w, faces = rot_and_correct_smplx_offset(pose_aa_w_i, trans_i, betas, final_Rt, get_verts=True)
    return pose_aa_w_i, trans_w_i


def prepare_multiperson_smpl_data(pose_aa_world, trans_world, betas, cam_world2cam, seq_name,
                                 to_floor=False, debug=0):

    pose_all, betas_all, trans_all, jts_img_all, joints_3d_all = [], [], [], [], []
    trans_delta = []
    for n in range(len(pose_aa_world)):
        # take the mesh to the floor; we then need to update camera params
        pose_aa_world_n, trans_world_n, betas_n = pose_aa_world[n], trans_world[n], betas[n]
        if to_floor:
            try:
                trans_world_n, trans_delta_n = trans_body_to_floor(pose_aa_world_n, trans_world_n, betas_n)
            except:
                print("Problems with SMPL mesh in trans_body_to_floor, skipping...")
                return None

        if debug:
            # save smpl mesh
            verts, faces = smpl_to_verts(pose_aa_world_n, trans_world_n, betas=betas_n[:, :10])
            out_path = f"inspect_out/phalp_process_world/{seq_name}/phalp_p{n}_000.ply"
            save_trimesh(verts[0, 0], faces, out_path)

        # PHALP estimates, need to do matching between phalp joints and the intial SLAHMR smpl bodies
        (_, joints_3d), _ = smpl_to_verts(pose_aa_world_n, trans_world_n, betas=betas_n[:, :10], return_joints=True)
        joints_3d_ref = joints_3d[0, 0, :25].detach().numpy()
        joints_camera = project_jts(cam_world2cam[0], joints_3d_ref)
        jts_img_all.append(joints_camera)
        pose_all.append(pose_aa_world_n)
        betas_all.append(betas_n)
        trans_all.append(trans_world_n)
        joints_3d_all.append(joints_3d)
        trans_delta.append(trans_delta_n)

    pose_aa_world = np.stack(pose_all)
    betas = np.stack(betas_all)
    trans = np.stack(trans_all)
    jts_img_all = np.stack(jts_img_all)
    joints_3d_all = np.stack(joints_3d_all)
    return_dict = {
        "pose_aa": pose_aa_world,
        "betas": betas,
        "trans": trans,
        "jts_img_all": jts_img_all,
        "joints_3d": joints_3d_all,
        "trans_delta_floor": trans_delta,
    }
    return return_dict


def assemble_camera(slahmr_dict, dyncam=False):
    rotation = slahmr_dict['rotation']
    translation = slahmr_dict['translation']
    focal_length = slahmr_dict['focal_length']
    camera_center = slahmr_dict['camera_center']
    T_w2c = slahmr_dict['T_w2c']

    ### Assemblle camera
    focal_length = focal_length.cpu()
    camera_center = camera_center.cpu()
    # for now one cam, todo change per-frame
    if dyncam:
        per_frame_cam = []
        for i in range(len(rotation)):
            full_R_sla = rotation[i]
            full_t_sla = translation[i]  # + np.array([0, -diff_trans, 0])
            K_sla = np.array([[focal_length[0, 0], 0, camera_center[0, 0]],
                              [0, focal_length[0, 1], camera_center[0, 1]],
                              [0., 0., 1.]])
            cam = {
                "full_R": full_R_sla,
                "full_t": full_t_sla,
                "K": K_sla,
                'img_w': 2 * camera_center[0, 0],
                'img_h': 2 * camera_center[0, 1],
                'scene_name': None,
                'T_w2c': T_w2c,
            }
            cam = to_numpy(cam)
            per_frame_cam.append(cam)
        cam = per_frame_cam
    else:
        rotation = rotation[0]
        translation = translation[0]
        full_R_sla = rotation # Tensor (3, 3)
        full_t_sla = translation #  + np.array([0, -diff_trans, 0]) # Tensor (3, )
        K_sla = np.array([[focal_length[0, 0], 0, camera_center[0, 0]],
                          [0, focal_length[0, 1], camera_center[0, 1]],
                          [0., 0., 1.]])
        cam = {
            "full_R": full_R_sla,
            "full_t": full_t_sla,
            "K": K_sla,
            'img_w': 2 * camera_center[0, 0],
            'img_h': 2 * camera_center[0, 1],
            'scene_name': None,
            'T_w2c': T_w2c
        }
        # make it a list
        cam = [to_numpy(cam)]
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
    trans_delta = new_trans - trans.numpy()
    return new_trans.copy(), trans_delta[0]


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


def get_pose_aa_from_slahmr_init_all(res_dict):
    # originally getting only the 1st frame info, but to override we need all
    B, T, _ = res_dict["pose_body"].shape
    body_pose_ = res_dict["pose_body"].cpu().numpy()
    betas = res_dict["betas"][:, None, :10].cpu().numpy()
    trans_orig = res_dict["trans"].cpu().numpy()
    global_orient = res_dict["root_orient"].cpu().numpy()
    body_pose = np.zeros([B, T, 69]).astype(np.float32)
    body_pose[:, :, :63] = body_pose_
    pose_aa = np.concatenate([global_orient, body_pose], axis=-1)
    return pose_aa, trans_orig, betas


def get_phalp_kpts(data):
    img_names = sorted(data.keys())
    all_joints = []
    is_complete = []
    for im_name in img_names:
        phalp_data = data[im_name]
        vitpose = phalp_data['vitpose']  # list of 2 with shape (25, 3)
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
    vis = j2d[ :, :, 2] > 0.1
    kpts_m = []
    for n, v in enumerate(vis):
        j2d_masked = j2d[n, v, :2]
        kpts_m.append(j2d_masked)
    return kpts_m


def fill_op_jts(op_kpts):
    op_kpts[:, :, 1] = (op_kpts[:, :, 2] + op_kpts[:, :, 5]) / 2
    op_kpts[:, :, 8] = (op_kpts[:, :, 9] + op_kpts[:, :, 12]) / 2
    return op_kpts


def make_video_w_skel(kpts_2d, seq_name, name="", data_name="viw"):
    im_path = f"/home/nugrinovic/code/NEURIPS_2023/slahmr/videos/{data_name}/images/{seq_name}"
    im_path = replace_slahmr_path(im_path)
    output_dir = f"/home/nugrinovic/code/NEURIPS_2023/slahmr/videos/{data_name}/vis_joints/{seq_name}"
    output_dir = replace_slahmr_path(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}{name}").mkdir(parents=True, exist_ok=True)
    for n, vit_jts in enumerate(tqdm(kpts_2d)):
        this_imgp = f"{im_path}/{n+1:06d}.jpg"
        img = read_image_PIL(this_imgp)
        j2d_all_i = fill_op_jts(vit_jts[..., :2][None])[0]
        img_w_jts = plot_skel_cv2(img, j2d_all_i)
        save_img(f"{output_dir}{name}/img_{n+1:04d}.png", img_w_jts)
    make_video(f"{output_dir}{name}/img_{n+1:04d}.png", ext="png", delete_imgs=True)


def get_img_w_pID(n, vit_jts, seq_name, t_id_all=None, data_name="viw", name=""):
    im_path = f"/home/nugrinovic/code/NEURIPS_2023/slahmr/videos/{data_name}/images/{seq_name}"
    im_path = replace_slahmr_path(im_path)
    this_imgp = f"{im_path}/{n + 1:06d}.jpg"
    img = read_image_PIL(this_imgp)
    vitpose_m = mask_joints_w_vis(vit_jts)
    try:
        bbox_vit = bbox_from_joints_several(vitpose_m)
    except:
        print(f"problem with shape of vitpose_m {vitpose_m[:, :, :2].shape}")
        print(f" vis is {vit_jts}")
    if t_id_all is None:
        img_w_boxes = plot_boxes_w_persID_cv2(img, bbox_vit, number_list=[1, 2], do_return=True)
    else:
        img_w_boxes = plot_boxes_w_persID_cv2(img, bbox_vit, number_list=t_id_all[n], do_return=True)
    return img_w_boxes
        

def make_video_w_pID(kpts_2d, seq_name, t_id_all=None, data_name="viw", name=""):
    im_path = f"/home/nugrinovic/code/NEURIPS_2023/slahmr/videos/{data_name}/images/{seq_name}"
    im_path = replace_slahmr_path(im_path)
    output_dir = f"/home/nugrinovic/code/NEURIPS_2023/slahmr/videos/{data_name}/vis_joints/{seq_name}"
    output_dir = replace_slahmr_path(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}_boxes{name}").mkdir(parents=True, exist_ok=True)
    for n, vit_jts in enumerate(tqdm(kpts_2d)):
        this_imgp = f"{im_path}/{n+1:06d}.jpg"
        img = read_image_PIL(this_imgp)
        vitpose_m = mask_joints_w_vis(vit_jts)

        try:
            bbox_vit = bbox_from_joints_several(vitpose_m)
        except:
            print(f"problem with shape of vitpose_m {vitpose_m[:, :, :2].shape}")
            print(f" vis is {vit_jts}")

        if t_id_all is None:
            img_w_boxes = plot_boxes_w_persID_cv2(img, bbox_vit, number_list=[1, 2], do_return=True)
        else:
            img_w_boxes = plot_boxes_w_persID_cv2(img, bbox_vit, number_list=t_id_all[n], do_return=True)

        save_img(f"{output_dir}_boxes{name}/img_{n+1:04d}.png", img_w_boxes)
    make_video(f"{output_dir}_boxes{name}/img_{n+1:04d}.png", ext="png", delete_imgs=True)


def get_phalp_kpts_simple(data, seq_name, ref_jts, data_name):
    smpl2op_map = smpl_to_openpose("smpl",
                                   use_hands=False,
                                   use_face=False,
                                   use_face_contour=False,
                                   openpose_format='coco25')

    img_names = sorted(data.keys())
    all_joints = []
    vitpose_orig = []
    for n, im_name in enumerate(img_names):
        phalp_data = data[im_name]
        vitpose = phalp_data['vitpose']  # list of 2 with shape (25, 3)
        # vitpose only has joints from 0 to 18, so 19 jts and does not have 1 (chest)
        # nor 8 (root), it does contain the nose at 0.
        # choose
        vitpose = np.stack(vitpose) # this is in OP format
        # vitpose_m = mask_joints_w_vis(vitpose)
        # bbox_vit = bbox_from_joints_several(vitpose_m[:, :, :2])
        # areas = compute_area_several(bbox_vit)
        vitpose_orig.append(vitpose)

        # choose which corresponds to this smpl person, do only first frame
        if n ==0 and data_name != 'hi4d':
            # ref_jts are in SMPL format
            smpl_2op_submap = smpl2op_map[smpl2op_map < 25]
            ref_jts_op = ref_jts[smpl_2op_submap]
            vitpose_14 = vitpose[:, :15]
            # make threshold more than 0.4
            confs_mask = vitpose_14[:, :, 2] > 0.4
            diff = confs_mask * np.abs(vitpose_14[:, :, :2] - ref_jts_op[None, :, :2]).mean(-1)
            diff = diff.mean(-1)
            idx_chosen = diff.argmin()
        elif data_name == 'hi4d':
            # ref_jts are in SMPL format
            smpl_2op_submap = smpl2op_map[smpl2op_map < 25]
            ref_jts_op = ref_jts[smpl_2op_submap]
            vitpose_14 = vitpose[:, :15]
            # make threshold more than 0.4
            confs_mask = vitpose_14[:, :, 2] > 0.4
            diff = confs_mask * np.abs(vitpose_14[:, :, :2] - ref_jts_op[None, :, :2]).mean(-1)
            diff = diff.mean(-1)
            idx_chosen = diff.argmin()

        vitpose_arr = np.zeros([1, 25, 3])
        try:
            vitpose_arr[0] = vitpose[idx_chosen]
        except:
            print(f"idx_chosen {idx_chosen} for {seq_name} out of range or iter {n}")

        confs = vitpose_arr[0, :, 2]
        confs = [c if c > 0.4 else 0. for c in confs]
        vitpose_arr[0, :, 2] = confs
        all_joints.append(vitpose_arr)

    op_kpts = np.stack(all_joints, 0)
    op_kpts[:, :, 1] = (op_kpts[:, :, 2] + op_kpts[:, :, 5]) / 2
    op_kpts[:, :, 8] = (op_kpts[:, :, 9] + op_kpts[:, :, 12]) / 2

    return op_kpts


def get_phalp_matched_kpts_simple_all(data, seq_name, ref_jts, data_name, subject, debug=False):
    smpl2op_map = smpl_to_openpose("smpl",
                                   use_hands=False,
                                   use_face=False,
                                   use_face_contour=False,
                                   openpose_format='coco25')
    img_names = sorted(data.keys())
    all_joints = []
    vitpose_orig = []
    t_id_all = []

    for n, im_name in enumerate(img_names):
        phalp_data = data[im_name]
        t_id = phalp_data['tid']
        vitpose = phalp_data['vitpose']  # list of 2 with shape (25, 3)
        t_id_all.append(t_id)
        vitpose_orig.append(np.stack(vitpose) )

    max_l = max([len(t) for t in t_id_all])

    # choose which corresponds to this smpl person, do only first frame
    vitpose = vitpose_orig[0]
    t_id = np.array(t_id_all[0])
    smpl_2op_submap = smpl2op_map[smpl2op_map < 25]
    ref_jts_op = ref_jts[:, smpl_2op_submap]
    vitpose_14 = vitpose[:, :15]
    # make threshold more than 0.4
    confs_mask = vitpose_14[:, :, 2] > 0.4
    vit_kpts = vitpose_14[:, :, :2]
    ref_kpts = ref_jts_op[:, :, :2]

    distances = []
    for vit, conf in zip(vit_kpts, confs_mask):
        distances.append((conf * np.linalg.norm(vit[None] - ref_kpts, axis=-1)).mean(-1))
    cost_matrix = np.stack(distances, 0)

    # Hungarian algo
    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=False)
    t_id_ord = t_id[row_ind] - 1
    col_ind = np.clip(col_ind, 0, len(t_id_ord)-1)
    t_id_chosen = t_id_ord[col_ind]
    vit2ref_map = {t_id_ord[i]: t_id_chosen[i] for i in range(len(t_id_ord))}


    for n, im_name in enumerate(img_names):
        phalp_data = data[im_name]
        vitpose = phalp_data['vitpose']  # list of 2 with shape (25, 3)
        t_id = phalp_data['tid']  # list of 2 with shape (25, 3)
        t_id = np.array(t_id) - 1
        vitpose = np.stack(vitpose) # this is in OP format
        # idx_chosen = ious.argmax()
        vitpose_arr = np.zeros([max_l, 25, 3])
        for i in range(len(vitpose)):
            try:
                vitpose_arr[vit2ref_map[t_id[i]]] = vitpose[i]
            except:
                # print(f"idx_chosen {i} for {seq_name} out of range or iter {n}")
                pass
        all_joints.append(vitpose_arr)
    op_kpts = np.stack(all_joints, 0)
    # am I using the conf values here, confs are taken directly from the vitpose output
    op_kpts[:, :, 1] = (op_kpts[:, :, 2] + op_kpts[:, :, 5]) / 2
    op_kpts[:, :, 8] = (op_kpts[:, :, 9] + op_kpts[:, :, 12]) / 2

    return op_kpts


def project_jts(cam, joints):
    """ this was used for convenience, please use perspective_projection() in any other case"""
    full_R_sla = cam["full_R"]
    full_t_sla = cam["full_t"]
    K_sla = cam["K"]
    jts_cam = (full_R_sla @ joints.T + full_t_sla[:, None]).T
    jts_img = (K_sla @ jts_cam.T).T
    jts_img = jts_img[:, :2] / jts_img[:, 2:]
    return jts_img


def get_files_from_dir(VIDEOS_ROOT, data_name, subject=None, return_dir=False):
    if data_name == 'chi3d':
        assert subject is not None, "subject must be provided for chi3d"
        vid_dir = f"{VIDEOS_ROOT}/{data_name}/train/{subject}/videos/50591643"
    else:
        vid_dir = f"{VIDEOS_ROOT}/{data_name}/{subject}/videos"
    video_files = sorted(Path(vid_dir).glob("*.mp4"))
    if return_dir:
        return video_files, vid_dir
    return video_files