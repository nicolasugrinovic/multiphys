import numpy as np
from pathlib import Path
import torch

import os.path as osp
from utils.pyquaternion import Quaternion as Q
from utils.misc import write_pickle
from utils.smpl import smpl_to_verts

from utils.net_utils import replace_emb_path
from utils.net_utils import get_hostname

from utils.misc import save_pointcloud
from utils.smpl import from_qpos_to_verts_w_model
from multiphys.process_data.preprocess.smpl_transforms import get_camera_transform
from multiphys.process_data.preprocess.smpl_transforms import rot_pose
from multiphys.process_data.preprocess.smpl_transforms import rot_and_correct_smplx_offset
from multiphys.process_data.preprocess.smpl_transforms import save_multi_mesh_sequence
from slahmr.geometry.camera import perspective_projection

from utils.misc import plot_joints_cv2
from utils.misc import read_image_PIL

hostname = get_hostname()


def bodies_to_world():
    """
    code intended to
    prepare_slahmr_results for computing metrics and comp to baseline,
    here I add the GT to the results dict from slahmr
    """

    print("Parsing data...")
    # load emb results to get the GT
    vid_dir = "/sailhome/nugrinov/code/CVPR_2024/slahmr_release/slahmr/videos/viw/videos"

    emb_res_dir = "/home/nugrinovic/code/CVPR_2024/EmbodiedPose/results/scene+/tcn_voxel_4_5_chi3d_multi_hum/results"
    # if it is orion it will change path if not it will stay the same
    emb_res_dir = replace_emb_path(emb_res_dir)
    path = f"{emb_res_dir}/chi3d/floor_2cams"
    vid_files = sorted(Path(vid_dir).glob("*.mp4"))

    # smpl_robot, humanoid, cc_cfg = load_humanoid()

    results_dict = {}
    # for seq_name in chi3d_names:
    for vfile in vid_files:
        # seq_name = 's03_Grab_10'
        vid_name = vfile.stem
        seq_name = vid_name
        print(vid_name)
        # skip_liest = ['011520_mpii_train', '011993_mpii_train']
        # if vid_name in skip_liest:
        #     continue

        results_dict[seq_name] = []
        res_dir = f"/sailhome/nugrinov/code/CVPR_2024/slahmr_release/slahmr/outputs/logs/viw-val/{vid_name}-all-shot-0-0-180"

        device = torch.device("cuda:0")
        try:
            res_dict = get_camera_transform(vid_name, res_dir)
        except:
            print(f"skipping {vid_name}, transforms data does not exist! generate in slahrm")
            continue
        # need to account for the plane inclination
        final_Rt = res_dict['final_Rt']#[0][None]
        jpos_pred = res_dict["joints"]
        jpos_pred = jpos_pred[:, :, :24]
        root_aa = res_dict['root_orient'].cpu()
        body_pose_aa = res_dict["pose_body"].cpu()
        B = root_aa.shape[0]
        seq_len = root_aa.shape[1]
        pose_aa = torch.cat([root_aa, body_pose_aa, torch.zeros(B, seq_len, 6).to(root_aa)], dim=-1)
        trans = res_dict["trans"].cpu()
        shape = res_dict["betas"][..., :10].cpu()
        mean_shape = shape[:, 0:10]  # .mean(axis=1)

        if 0:
            # inspect if these project well to the img
            imgs_dir = f"/sailhome/nugrinov/code/CVPR_2024/slahmr_release/slahmr/videos/viw/images/{vid_name}"
            img_files = sorted(Path(imgs_dir).glob("*.jpg"))

            focal_length = res_dict['intrins'][:2][None]
            camera_center = res_dict['intrins'][2:][None]

            T_c2w = res_dict["cameras"]["src_cam"]
            T_w2c = torch.inverse(T_c2w)
            rotation = T_w2c[:, :3, :3]
            translation = T_w2c[:, :3, 3]
            j2d_all = []
            for p_id in range(len(jpos_pred)):
                jts_pred_img = perspective_projection(jpos_pred[p_id], focal_length, camera_center, rotation, translation)
                j2d_all.append(jts_pred_img)
            j2d_all = torch.stack(j2d_all, dim=0).cpu()
            # read image
            idx = 0
            img = read_image_PIL(img_files[idx])
            plot_joints_cv2(img, j2d_all[:, idx], show=True, with_text=False, sc=3)

        idx = 0
        verts_orig_all = []
        verts_w_all = []
        verts_w_rot_all = []
        all_verts_i = []
        all_verts_gt_i = []
        all_verts_r_i = []
        all_joints_w = []
        all_joints_w_rot = []
        for p_id in range(len(jpos_pred)):
            jpos_pred_world = (final_Rt[:, :3, :3] @ jpos_pred[p_id].permute(0, 2, 1)).permute(0, 2, 1) + final_Rt[:, None, :3, 3]
            # NOTE: for some reason we have to rotate 180 over z-axis (check why).
            T = Q(axis=[0, 0, 1], angle=np.pi).transformation_matrix.astype(np.float32)
            Tz180 = torch.from_numpy(T).to(jpos_pred.device)
            jpos_pred_world_rot = (Tz180[None, :3, :3] @ jpos_pred_world.permute(0, 2, 1)).permute(0, 2, 1)
            all_joints_w.append(jpos_pred_world)
            all_joints_w_rot.append(jpos_pred_world_rot)


            if 1:
                # save_pointcloud(jpos_pred[p_id, 0], f"inspect_out/slahmr/prep_eval/joints/{seq_name}/jpos_pred_sla_{p_id}.ply")
                # save_pointcloud(jpos_pred_w_rot[0], f"inspect_out/eval/joints/jpos_pred_w_rot_sla_{p_id}.ply")
                save_pointcloud(jpos_pred_world[idx],
                                f"inspect_out/slahmr/viw_to_world/{seq_name}/joints/jpos_pred_world_{idx}_{p_id}.ply")
                save_pointcloud(jpos_pred_world_rot[idx],
                                f"inspect_out/slahmr/viw_to_world/{seq_name}/joints/jpos_pred_world_rot_{idx}_{p_id}.ply")
                # save_pointcloud(gt_jts[idx], f"inspect_out/slahmr/prep_eval/{seq_name}/joints/gt_jts_{idx}_{p_id}.ply")

            gender = "neutral"
            # smpl_robot.load_from_skeleton(mean_shape[p_id, None], gender=[0], objs_info=None)
            # model = mujoco_py.load_model_from_xml(smpl_robot.export_xml_string().decode("utf-8"))

            # transform to world coords with camera and ground plane orientation, final_Rt contains both
            betas = mean_shape[None, p_id, :10].to(device)
            pose_aa_i = pose_aa[p_id]
            trans_i = trans[p_id].clone().to(device)
            verts_orig, faces = smpl_to_verts(pose_aa_i, trans_i.float(), betas=betas[:1])
            verts_orig_all.append(verts_orig)

            # transform to world
            pose_aa_w_i = rot_pose(pose_aa_i.cpu(), final_Rt.cpu()).to(device)
            trans_w_i, verts_i_w, faces = rot_and_correct_smplx_offset(pose_aa_w_i, trans_i, betas, final_Rt, get_verts=True)
            # qpos = smpl_to_qpose_torch(pose_aa_i, model, trans=trans_i, count_offset=True)
            verts_w_all.append(verts_i_w)

            # rotate 180 over z-axis
            # pose_aa_w_rot_i = rot_pose(pose_aa_w_i.cpu(), Tz180.cpu()).to(trans_i)
            # trans_w_rot_i, verts_w_rot_i, faces = rot_and_correct_smplx_offset(pose_aa_w_rot_i, trans_w_i, betas, Tz180, get_verts=True)
            # qpos_rot = smpl_to_qpose_torch(pose_aa_i_rot.cuda(), model, trans=trans_i_rot, count_offset=True)
            # qpos_rot = qpos_rot.cpu().numpy()

            gt_data = {}
            # verts_w_rot_all.append(verts_w_rot_i)

            # check
            if 0:
                # todo funtion changed, must update to use
                verts, faces = from_qpos_to_verts_w_model(qpos.cpu().numpy(), model, cc_cfg)
                all_verts_i.append(verts)

                verts, faces = from_qpos_to_verts_w_model(qpos_gt, model, cc_cfg)
                all_verts_gt_i.append(verts[:, :seq_len])

                verts, faces = from_qpos_to_verts_w_model(qpos_rot, model, cc_cfg)
                all_verts_r_i.append(verts[:, :seq_len])

            # joints
            gt_data['pred_jpos'] = jpos_pred_world_rot.cpu().numpy()
            # other
            gt_data['percent'] = 1.0
            gt_data['fail_safe'] = False
            results_dict[seq_name].append(gt_data)

        if 1:
            output_dir = f"inspect_out/slahmr/viw_to_world/{seq_name}/meshes"

            mesh_folder = osp.join(output_dir, f"smpl_meshes_orig")
            save_multi_mesh_sequence(verts_orig_all, faces, mesh_folder, name='_smpl_orig', ds=4)

            mesh_folder = osp.join(output_dir, f"smpl_meshes_w")
            # save_multi_mesh_sequnce(verts_w_all, faces, mesh_folder, name='_smpl_w', ds=4)
            save_multi_mesh_sequence(verts_w_all, faces, mesh_folder, name='_smpl_w', ds=4)

            # mesh_folder = osp.join(output_dir, f"smpl_meshes_w_rot")
            # save_multi_mesh_sequnce(verts_w_rot_all, faces, mesh_folder, name='_smpl_w_rot', debug_idx=idx)


            # mesh_folder = osp.join(output_dir, f"qpos_meshes")
            # save_multi_mesh_sequnce(all_verts_i, faces, mesh_folder, name='_qpos_pred', debug_idx=idx)
            #
            # mesh_folder = osp.join(output_dir, f"qpos_rot_meshes")
            # save_multi_mesh_sequnce(all_verts_r_i, faces, mesh_folder, name='_qpos_rot', debug_idx=idx)
            print("saved")

        all_joints_w = torch.stack(all_joints_w, dim=0).cpu()
        all_joints_w_rot = torch.stack(all_joints_w_rot, dim=0).cpu()
        # save
        out_path = f"inspect_out/slahmr/viw_to_world/{seq_name}/joints/joints_w.pkl"
        write_pickle(all_joints_w, out_path)
        out_path = f"inspect_out/slahmr/viw_to_world/{seq_name}/joints/joints_w_rot.pkl"
        write_pickle(all_joints_w_rot, out_path)



if __name__ == "__main__":
    bodies_to_world()

