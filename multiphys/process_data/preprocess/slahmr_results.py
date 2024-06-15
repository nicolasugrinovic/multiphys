import numpy as np
from slahmr.output import get_results_paths, load_result
from embpose_util.tensor import get_device, move_to, detach_all, to_torch
from utils.misc import read_npy
from utils.misc import read_json
from utils.misc import save_trimesh

from slahmr.vis_output import prep_result_vis
from utils.smpl import smpl_to_verts
import torch
from scipy.spatial.transform import Rotation as sRot
from utils.net_utils import replace_orion_root
from slahmr.geometry import camera as cam_util

from pyquaternion import Quaternion as Q
from math import pi

def dict_to_np(d):
    for k, v in d.items():
        d[k] = np.array(v)
    return d

def main():
    res_dir = "/mnt/Data/nugrinovic/code/NEURIPS_2023/slahmr/outputs/logs/chi3d-val/Grab_1-all-shot-0-0-180/motion_chunks"
    res_dir = "/sailhome/nugrinov/code/CVPR_2024/slahmr_release/slahmr/outputs/logs/chi3d-val/Grab_1-all-shot-0-0-180/motion_chunks"
    res_path_dict = get_results_paths(res_dir)
    # load last iter
    it = sorted(res_path_dict.keys())[-1]
    res_dict = load_result(res_path_dict[it])["world"]

    # vid_name = "Grab_1-all-shot-0-0-180"
    # res_dict = move_to(res_dict, device)
    # cam_dir = f"/home/nugrinovic/code/NEURIPS_2023/slahmr/videos/viw/slahmr/cameras/{vid_name}/shot-0"
    # cameras = f"{cam_dir}/cameras.npz"
    #
    # cam_dir = "/home/nugrinovic/code/NEURIPS_2023/slahmr/outputs/logs/chi3d-val/Grab_1-all-shot-0-0-180/cameras.json"
    # cams = read_json(cam_dir)
    # cams = dict_to_np(cams)
    #
    # # ['height', 'width', 'focal', 'intrins', 'w2c']
    # slam_camera = read_npy(cameras)
    # height = slam_camera['height'].item()
    # width = slam_camera['width'].item() # 640
    # focal = slam_camera['focal'].item()
    # intrins = slam_camera['intrins'][0]
    # w2c = slam_camera['w2c'][0]
    # # ['cam_R', 'cam_t', 'intrins', 'static']
    # cam_data = {}
    # cam_data['cam_R'] = w2c[:3, :3]
    # cam_data['cam_t'] = w2c[:3, 3]
    # cam_data['intrins'] = intrins
    # cam_data['static'] = False
    
    root_orient = res_dict["root_orient"]
    pose_body = res_dict["pose_body"]
    trans = res_dict["trans"]
    B, seq_len, _ = pose_body.shape
    pose_body_72 = torch.zeros((B, seq_len, 72))
    pose_body_72[:, :, :3] = root_orient
    pose_body_72[:, :, 3:66] = pose_body
    trans_ = trans[0]
    pose_body_72 = pose_body_72[0]
    (verts, joints), faces = smpl_to_verts(pose_body_72, trans_, betas=None, return_joints=True)

    world_smpl = {}
    world_smpl["joints"] = joints

    vis_mask = np.ones((seq_len))
    scene_dict = prep_result_vis(
        world_smpl,
        res_dict,
        vis_mask,
        # obs_data["track_id"],
    )

    Rt_gnd = scene_dict["ground"]
    T_c2w = scene_dict["cameras"]["src_cam"] # this is T_c2w

    # transform the bodies
    samp_verts = verts[0, 0]
    save_trimesh(samp_verts, faces, "inspect_out/slahmr/camera/mesh_orig.ply")
    verts_world = (T_c2w[0, :3, :3] @ samp_verts.permute(1, 0)).permute(1, 0) + T_c2w[:1, :3, 3]
    save_trimesh(verts_world, faces, "inspect_out/slahmr/camera/verts_world.ply")

    Rt_gnd_inv = torch.linalg.inv(Rt_gnd)

    verts_world_gnd = (Rt_gnd_inv[:3, :3] @ verts_world.permute(1, 0)).permute(1, 0) + Rt_gnd_inv[:3, 3]
    save_trimesh(verts_world_gnd, faces, "inspect_out/slahmr/camera/verts_world_gnd.ply")
    # T_c2w[:1, :3, 3].shape
    rot_90 = sRot.from_euler("xyz", np.array([-np.pi/2, 0, 0])).as_matrix() # (3, 3)
    rot_90 = torch.tensor(rot_90).float().to(T_c2w.device)
    verts_world_gnd_rot = (rot_90 @ verts_world_gnd.permute(1, 0)).permute(1, 0)
    save_trimesh(verts_world_gnd_rot, faces, "inspect_out/slahmr/camera/verts_world_gnd_rot.ply")

    Txm90 = Q(axis=[1, 0, 0], angle=-np.pi/2).transformation_matrix.astype(np.float32)
    Txm90 = torch.tensor(Txm90).float().to(T_c2w.device)

    final_Rt = Txm90 @ Rt_gnd_inv @ T_c2w[0]
    verts_world_emb = (final_Rt[:3, :3] @ samp_verts.permute(1, 0)).permute(1, 0) + final_Rt[:3, 3]
    save_trimesh(verts_world_emb, faces, "inspect_out/slahmr/camera/verts_world_emb.ply")



def get_world_smpl(res_dict):
    world_smpl = {}
    root_orient = res_dict["root_orient"]
    pose_body = res_dict["pose_body"]
    trans = res_dict["trans"]
    B, seq_len, _ = pose_body.shape
    pose_body_72 = torch.zeros((B, seq_len, 72))
    pose_body_72[:, :, :3] = root_orient
    pose_body_72[:, :, 3:66] = pose_body
    (verts, joints), faces = smpl_to_verts(pose_body_72[0], trans[0], betas=None, return_joints=True)
    world_smpl["joints"] = joints
    world_smpl["vertices"] = verts
    return world_smpl, seq_len


def get_cameras_slahmr(
    res_dir,
):
    """ 
    a better function for doing the same as above:
    """
    # res_dir = "/mnt/Data/nugrinovic/code/NEURIPS_2023/slahmr/outputs/logs/chi3d-val/Grab_1-all-shot-0-0-180/motion_chunks"
    res_dir = "/sailhome/nugrinov/code/CVPR_2024/slahmr_release/slahmr/outputs/logs/chi3d-val/Grab_1-all-shot-0-0-180/motion_chunks"
    
    res_path_dict = get_results_paths(res_dir)
    # load last iter, this is the final slahmr result
    it = sorted(res_path_dict.keys())[-1]
    res_dict = load_result(res_path_dict[it])["world"]
    
    world_smpl, seq_len = get_world_smpl(res_dict)

    vis_mask = np.ones((seq_len))
    scene_dict = prep_result_vis(
        world_smpl,
        res_dict,
        vis_mask,
        # obs_data["track_id"],
    )

    Rt_gnd = scene_dict["ground"]
    T_c2w = scene_dict["cameras"]["src_cam"]  # this is T_c2w

    # transform the bodies
    # samp_verts = verts[0, 0]
    # save_trimesh(samp_verts, faces, "inspect_out/slahmr/camera/mesh_orig.ply")
    # verts_world = (T_c2w[0, :3, :3] @ samp_verts.permute(1, 0)).permute(1, 0) + T_c2w[:1, :3, 3]
    # save_trimesh(verts_world, faces, "inspect_out/slahmr/camera/verts_world.ply")
    Rt_gnd_inv = torch.linalg.inv(Rt_gnd)
    # verts_world_gnd = (Rt_gnd_inv[:3, :3] @ verts_world.permute(1, 0)).permute(1, 0) + Rt_gnd_inv[:3, 3]
    # save_trimesh(verts_world_gnd, faces, "inspect_out/slahmr/camera/verts_world_gnd.ply")
    # T_c2w[:1, :3, 3].shape
    rot_90 = sRot.from_euler("xyz", np.array([-np.pi / 2, 0, 0])).as_matrix()  # (3, 3)
    rot_90 = torch.tensor(rot_90).float().to(T_c2w.device)
    # verts_world_gnd_rot = (rot_90 @ verts_world_gnd.permute(1, 0)).permute(1, 0)
    # save_trimesh(verts_world_gnd_rot, faces, "inspect_out/slahmr/camera/verts_world_gnd_rot.ply")
    Txm90 = Q(axis=[1, 0, 0], angle=-np.pi/2).transformation_matrix.astype(np.float32)
    Txm90 = torch.tensor(Txm90).float().to(T_c2w.device)
    final_Rt = Txm90 @ Rt_gnd_inv @ T_c2w[0]
    # verts_world_emb = (final_Rt[:3, :3] @ samp_verts.permute(1, 0)).permute(1, 0) + final_Rt[:3, 3]
    # save_trimesh(verts_world_emb, faces, "inspect_out/slahmr/camera/verts_world_emb.ply")


    
    

if __name__ == "__main__":
    main()
