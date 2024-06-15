from pathlib import Path

import torch
from tqdm import tqdm

from multiphys.process_data.preprocess.smpl_transforms import get_camera_transform
from slahmr.geometry.camera import perspective_projection
from utils.misc import plot_joints_cv2
from utils.misc import read_image_PIL
from utils.misc import read_pickle
from utils.misc import save_img
from utils.net_utils import get_hostname
from utils.video import make_video

hostname = get_hostname()


def get_slahmr_emb2cam_transform(SLA_ROOT, data_name, vid_name, rot_180z=False, subject=None):
    if subject is not None:
        res_dir = f"{SLA_ROOT}/outputs/logs/{data_name}-val/{subject}/{vid_name}-all-shot-0-0-180"
    else:
        res_dir = f"{SLA_ROOT}/outputs/logs/{data_name}-val/{vid_name}-all-shot-0-0-180"

    # we dont want subdir here
    # if sub_dir is not None:
    #     res_dir = f"{res_dir}/{sub_dir}"
    # T = Q(axis=[0, 0, -1], angle=np.pi).transformation_matrix.astype(np.float32)
    # Tz180_inv = torch.from_numpy(T)

    res_dict = get_camera_transform(vid_name, res_dir, rot_180z)
    final_Rt = res_dict['final_Rt']
    # project joints back to image
    final_Rt_inv = torch.inverse(final_Rt).cpu()
    focal_length = res_dict['intrins'][:2][None]
    camera_center = res_dict['intrins'][2:][None]

    T_c2w = res_dict["cameras"]["src_cam"]
    T_w2c_orig = torch.inverse(T_c2w).cpu()

    # T_w2c = T_w2c @ final_Rt_inv @ Tz180_inv
    # no need to rotate by Tz180_inv, already included in final_Rt
    T_w2c = T_w2c_orig @ final_Rt_inv

    rotation = T_w2c[:, :3, :3]
    translation = T_w2c[:, :3, 3]
    res_dict['res_dir'] = res_dir
    return rotation, translation, focal_length, camera_center, res_dict, T_w2c_orig, final_Rt, final_Rt_inv


def inspect_dynamic_cam_projection(vid_name, inspect=False):
    res_dir = f"/sailhome/nugrinov/code/CVPR_2024/slahmr_release/slahmr/outputs/logs/viw-val/{vid_name}-all-shot-0-0-180"
    device = torch.device("cuda:0")
    try:
        # res_dict = get_camera_transform(vid_name, res_dir)
        rotation, translation, focal_length, camera_center, res_dict = get_slahmr_emb2cam_transform(vid_name)
    except:
        print(f"skipping {vid_name}, transforms data does not exist! generate in slahrm")
        return 0

    data_path = f"inspect_out/slahmr/viw_to_world/{vid_name}/joints/joints_w.pkl"
    try:
        j3d = read_pickle(data_path)
    except:
        print(f"skipping {vid_name}, transforms data does not exist! generate in slahrm")
        return 0
    # final_Rt = res_dict['final_Rt']
    # # project joints back to image
    # final_Rt_inv = torch.inverse(final_Rt).cpu()
    # focal_length = res_dict['intrins'][:2][None]
    # camera_center = res_dict['intrins'][2:][None]
    #
    # T_c2w = res_dict["cameras"]["src_cam"]
    # T_w2c = torch.inverse(T_c2w).cpu()
    # T_w2c = T_w2c @ final_Rt_inv
    # rotation = T_w2c[:, :3, :3]
    # translation = T_w2c[:, :3, 3]

    if inspect:
        j2d_all = []
        for p_id in range(len(j3d)):
            jts_pred_img = perspective_projection(j3d[p_id], focal_length, camera_center, rotation, translation)
            j2d_all.append(jts_pred_img)
        j2d_all = torch.stack(j2d_all, dim=0)
        imgs_dir = f"/sailhome/nugrinov/code/CVPR_2024/slahmr_release/slahmr/videos/viw/images/{vid_name}"
        img_files = sorted(Path(imgs_dir).glob("*.jpg"))
        # read image
        output_dir = f"inspect_out/slahmr/viw_to_world/{vid_name}/video_kpts"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for idx in tqdm(range(len(img_files))):
            img = read_image_PIL(img_files[idx])
            img_w_jts = plot_joints_cv2(img, j2d_all[:, idx], show=False, with_text=False, sc=3)
            save_img( f"{output_dir}/img_{idx:04d}.png", img_w_jts)
        # save video
        make_video(f"{output_dir}/img_{idx:04d}.png")


def main():
    """
    code intended to
    prepare_slahmr_results for computing metrics and comp to baseline,
    here I add the GT to the results dict from slahmr
    """
    print("Parsing data...")
    # load emb results to get the GT
    vid_dir = "/sailhome/nugrinov/code/CVPR_2024/slahmr_release/slahmr/videos/viw/videos"
    # if it is orion it will change path if not it will stay the same
    vid_files = sorted(Path(vid_dir).glob("*.mp4"))
    for vfile in vid_files:
        # seq_name = 's03_Grab_10'
        vid_name = vfile.stem
        print(vid_name)
        inspect_dynamic_cam_projection(vid_name, inspect=True)


if __name__ == "__main__":
    main()