import numpy as np
from pathlib import Path
from utils.misc import read_image_PIL
from utils.misc import plot_skel_cv2
import os.path as osp
from tqdm import tqdm
from utils.misc import save_img
from utils.misc import plot
import cv2
from utils.net_utils import get_hostname
from utils.video import add_captions_to_frame
import os

def video_from_images(out_path):
    from pathlib import Path
    out_path = Path(out_path)
    os.system(
        f"ffmpeg -framerate 30 -pattern_type glob -i '{out_path.parent}/*.png' "
        f"-c:v libx264 -vf fps=30 -pix_fmt yuv420p {out_path.parent}.mp4 -y")
    os.system(f"rm -rf {out_path.parent}")


def make_video(out_path, ext="png"):
    import os
    # out_path = Path(f"inspect_out/chi3d/proj2d/xxx.png")
    out_path = Path(out_path)
    cmd = f"ffmpeg -framerate 30 -pattern_type glob -i '{out_path.parent}/*.{ext}' -c:v libx264 -vf fps=30 -pix_fmt yuv420p {out_path.parent}.mp4 -y"
    print(cmd)
    os.system(cmd)
    # if delete_imgs:
    os.system(f"rm -rf {out_path.parent}")

def visu_estimates_gen(data_name, seq_name, subj_name, output_dir, joints2d_all, pred_joints2d_all,
                       j_type="OP", name='', debug=False):

    print("EXTRACTING original video frames ")
    img_files = extract_video_frames(data_name, seq_name, subj_name)

    j2_len = len(joints2d_all)
    out_joint_imgs = osp.join(output_dir, f"reproj_2d_{seq_name}_{name}")

    Path(out_joint_imgs).mkdir(parents=True, exist_ok=True)
    print('Saving joint images...')
    for n, file in enumerate(tqdm(img_files[:j2_len])):
        img = read_image_PIL(file)
        j2d = joints2d_all[n]
        pred_j2d = pred_joints2d_all[n]
        # if img is too big, resize it, otherwise it kills the process
        h, w, _ = img.shape
        if h == 2048 and w==2048:
            img = cv2.resize(img, (h//2, h//2))
            j2d[..., :2] = j2d[..., :2] / 2
            pred_j2d[..., :2] = pred_j2d[..., :2] / 2
        # this plot_skel_cv2 works for multi-person, just pass an array with batch dim = num people
        img_w_kpts2d = plot_skel_cv2(img, j2d[:, :, :2], alpha=0.6)
        img_w_kpts2d = plot_skel_cv2(img_w_kpts2d, pred_j2d, type=j_type, alpha=0.6, all_yellow=True)
        outp = Path(f"{out_joint_imgs}/{n:06d}.png")
        outp.parent.mkdir(parents=True, exist_ok=True)
        if debug:
            plot(img_w_kpts2d)
            return

        save_img(outp, img_w_kpts2d)

    make_video(outp)  # img here are always png
    return out_joint_imgs


def visu_estimates(cfg, agent, take_key, frames, output_dir, cam_num, pname):

    if cfg.subject is None:
        sub_name = take_key.split("_")[0]
    else:
        sub_name = cfg.subject
    print("EXTRACTING original video frames ")
    img_files = extract_video_frames(cfg.data_name, take_key, sub_name)

    joints2d_all = []
    pred_joints2d_all = []
    for n in range(agent.env.num_agents):
        joints2d = agent.data_loader.data_raw[n][take_key]["joints2d"]
        # por alguna razon que no se, hay que descartar los primeros 2 frames aqui
        joints2d_all.append(joints2d[1:])
        pred_joints2d = agent.env.pred_joints2d[n][1:]
        pred_joints2d_all.append(np.concatenate(pred_joints2d, 0))

    agent.env.pred_joints2d = [[] for _ in range(agent.env.num_agents)]
    joints2d_all = np.stack(joints2d_all, 1)
    pred_joints2d_all = np.concatenate(pred_joints2d_all, 1)
    j2_len = len(frames)

    out_joint_imgs = osp.join(output_dir, f"{cam_num}_results_w_2d_{pname}")

    Path(out_joint_imgs).mkdir(parents=True, exist_ok=True)
    print('Saving joint images...')
    for n, file in enumerate(tqdm(img_files[:j2_len])):
        img = read_image_PIL(file)
        img_orig = img.copy()
        # plot(img)
        # if data_name == "prox":
        #     # flip image horizontally
        #     img = cv2.flip(img, 1)
        j2d = joints2d_all[n]
        pred_j2d = pred_joints2d_all[n]

        # if img is too big, resize it, otherwise it kills the process
        h, w, _ = img.shape
        if h == 2048 and w==2048:
            img = cv2.resize(img, (h//2, h//2))
            img_orig = img.copy()
            j2d[..., :2] = j2d[..., :2] / 2
            pred_j2d[..., :2] = pred_j2d[..., :2] / 2

        black_bg = True
        alpha = 0.7
        offset = -20
        fontSize = 2.0
        thickness = 6
        img_orig = add_captions_to_frame(img_orig, 'Input Video', alpha=alpha, black_bg=black_bg,
                                         offset=offset, fontSize=fontSize, thickness=thickness)
        # plot(img_orig)
        # this plot_skel_cv2 works for multi-person, just pass an array with batch dim = num people
        img_w_kpts2d = plot_skel_cv2(img, j2d)
        if agent.cfg.vis_pred_kpts:
            img_w_kpts2d = plot_skel_cv2(img_w_kpts2d, pred_j2d, type="emb", alpha=0.6)
        outp = Path(f"{out_joint_imgs}/{n:06d}.png")
        outp.parent.mkdir(parents=True, exist_ok=True)
        frame = frames[n]

        img_w_kpts2d = add_captions_to_frame(img_w_kpts2d[..., :3], 'ViT Pose', alpha=alpha, black_bg=black_bg,
                                         offset=offset, fontSize=fontSize, thickness=thickness)
        frame = add_captions_to_frame(frame, 'MultiPhys', alpha=alpha, black_bg=black_bg,
                                         offset=offset, fontSize=fontSize, thickness=thickness)

        img_cat = np.concatenate([img_orig, img_w_kpts2d, frame], axis=1)
        # plot(img_w_kpts2d)
        # plot(frame)
        # plot(img_cat)
        # if cfg.data_name == "prox":
        #     scale_percent = 30  # percent of original size
        #     width = int(img_cat.shape[1] * scale_percent / 100)
        #     height = int(img_cat.shape[0] * scale_percent / 100)
        #     dim = (width, height)
        #     # resize image
        #     img_cat = cv2.resize(img_cat, dim, interpolation=cv2.INTER_AREA)
        save_img(outp, img_cat)

    make_video(outp)  # img here are always png
    return out_joint_imgs


def extract_video_frames(data_name, take_key, sub_name):

    if data_name == "chi3d":
        img_folder = f"./data/videos/chi3d/train/{sub_name}/images/{take_key[4:]}"
        img_files = sorted(list(Path(img_folder).glob("*.jpg")))
    elif data_name == "videos_in_wild":
        img_folder = f"./data/videos/viw/images/{take_key}"
        img_files = sorted(list(Path(img_folder).glob("*.jpg")))
    elif data_name == "hi4d":
        img_folder = f"./data/videos/hi4d/images/{take_key}"
        img_files = sorted(list(Path(img_folder).glob("*.jpg")))
    elif data_name == "expi":
        new_key = take_key[6:]
        img_folder = f"./data/videos/{data_name}/{sub_name}/images/{new_key}"
        img_files = sorted(list(Path(img_folder).glob("*.jpg")))
    elif data_name == "shorts":
        img_folder = f"./data/videos/shorts/{sub_name}/images/{take_key}"
        img_files = sorted(list(Path(img_folder).glob("*.jpg")))
    else:
        print(f"Not implemented!! -- data_name {data_name} not implemented!!")
        raise NotImplementedError

    print(f"Data name is {data_name}")
    print(f"Found {len(img_files)} images -- in img folder {img_folder}")

    return img_files


# def extract_video_frames(data_name, take_key, sub_name):
#     hostname = get_hostname()
#     if "oriong" in hostname:
#         SLA_ROOT = "/sailhome/nugrinov/code/CVPR_2024/slahmr_release/slahmr"
#     else:
#         SLA_ROOT = "/home/nugrinovic/code/CVPR_2024/slahmr_release/slahmr"
#
#     if data_name == "chi3d":
#         img_folder = f"{SLA_ROOT}/videos/chi3d/train/{sub_name}/images/{take_key[4:]}"
#         img_files = sorted(list(Path(img_folder).glob("*.jpg")))
#     elif data_name == "prox":
#         prox_recs = "/home/nugrinovic/code/CVPR_2024/EmbodiedPose/data/prox/recordings"
#         img_files = sorted(list(Path(f"{prox_recs}/{take_key}/Color").glob("*.jpg")))
#     elif data_name == "videos_in_wild":
#         img_folder = f"{SLA_ROOT}/videos/viw/images/{take_key}"
#         img_files = sorted(list(Path(img_folder).glob("*.jpg")))
#     elif data_name == "chi3d_slahmr":
#         vname = take_key.replace(" ", "_")
#         vname = vname[4:]
#         img_folder = f"{SLA_ROOT}/videos/chi3d/train/{sub_name}/images/{vname}"
#         img_files = sorted(list(Path(img_folder).glob("*.jpg")))
#     elif data_name == "hi4d":
#         img_folder = f"{SLA_ROOT}/videos/hi4d/images/{take_key}"
#         img_files = sorted(list(Path(img_folder).glob("*.jpg")))
#     elif data_name == "expi":
#         new_key = take_key[6:]
#         img_folder = f"{SLA_ROOT}/videos/{data_name}/{sub_name}/images/{new_key}"
#         img_files = sorted(list(Path(img_folder).glob("*.jpg")))
#     elif data_name == "shorts":
#         img_folder = f"{SLA_ROOT}/videos/shorts/{sub_name}/images/{take_key}"
#         img_files = sorted(list(Path(img_folder).glob("*.jpg")))
#     else:
#         print(f"Not implemented!! -- data_name {data_name} not implemented!!")
#         raise NotImplementedError
#
#     print(f"Data name is {data_name}")
#     print(f"Found {len(img_files)} images -- in img folder {img_folder}")
#
#     return img_files