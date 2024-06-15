from pathlib import Path

import cv2

from utils.misc import plot_joints_cv2
from utils.misc import read_pickle
from utils.video import make_video
import joblib
import numpy as np
from utils.misc import read_pickle
from utils.misc import read_json
from utils.misc import read_image_PIL
from utils.misc import plot
from utils.misc import plot_joints_cv2
import joblib
from pathlib import Path
import cv2
from utils.misc import save_img
from tqdm import tqdm
import os
from utils.misc import plot_skel_cv2

def main():
    file_path = Path("sample_data/chi3d/thirdeye_clip_chi3d_s02_embodied_cam2w_p2_floor.pkl")
    file_path = Path("sample_data/chi3d/thirdeye_clip_chi3d_s02_embodied_cam2w_p2_floor_smooth2d.pkl")

    out_path = Path(f"inspect_out/chi3d/pre_processed")

    emb_data_p = joblib.load(file_path)

    seq_name = "s02_Grab_1"
    fname = file_path.stem.split("_")[5:]
    fname = "_".join(fname)
    emb_data_p = emb_data_p[seq_name]
    joints2d = emb_data_p["joints2d"]
    dlength = len(joints2d)

    videos_path = "data/chi3d/train/s02/videos/50591643/Grab 1.mp4"
    cap = cv2.VideoCapture(videos_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    imgs = []
    i = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        if i >= dlength:
            break
        orig_img = frame
        j2d = joints2d[i]
        img_w_kpts2d = plot_joints_cv2(orig_img, j2d[None, :, :2], with_text=False, sc=2, show=False)
        outp = out_path / Path(f"{fname}/{i:04d}.png")
        outp.parent.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str(outp), img_w_kpts2d)
        imgs.append(img_w_kpts2d)
        i += 1

    make_video(outp)


def plot_skeleton_2d():
    file_path = Path("sample_data/chi3d/thirdeye_clip_chi3d_s02_embodied_cam2w_p2_floor.pkl")
    file_path = Path("sample_data/chi3d/thirdeye_clip_chi3d_s02_embodied_cam2w_p2_floor_smooth2d.pkl")
    out_path = Path(f"inspect_out/chi3d/pre_processed")
    emb_data_p = joblib.load(file_path)
    seq_name = "s02_Grab_1"
    fname = file_path.stem.split("_")[5:]
    fname = "_".join(fname)
    emb_data_p = emb_data_p[seq_name]
    joints2d = emb_data_p["joints2d"]
    dlength = len(joints2d)

    videos_path = "data/chi3d/train/s02/videos/50591643/Grab 1.mp4"
    cap = cv2.VideoCapture(videos_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    imgs = []
    i = 0
    while (cap.isOpened()):
        ret, img = cap.read()
        if ret == False:
            break
        if i >= dlength:
            break

        j2d = joints2d[i]
        img_w_kpts2d = plot_skel_cv2(img, j2d)
        outp = out_path / Path(f"{fname}/{i:04d}.png")
        outp.parent.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str(outp), img_w_kpts2d)
        imgs.append(img_w_kpts2d)
        i += 1

    make_video(outp)


def inspect_dekr_detections():
    file_path = Path("/home/nugrinovic/code/CVPR_2024/DEKR/output/coco_format/chi3d/s02/Grab_1/preds.pkl")
    out_path = Path(f"inspect_out/chi3d/dekr_detections")
    emb_data_p = joblib.load(file_path)
    seq_n = file_path.parts[9]
    act_name = file_path.parts[10]
    joints2d = emb_data_p
    dlength = len(joints2d)

    videos_path = "data/chi3d/train/s02/videos/50591643/Grab 1.mp4"
    cap = cv2.VideoCapture(videos_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(length)

    imgs = []
    i = 0
    with tqdm(total=length) as pbar:
        while (cap.isOpened()):
            ret, img = cap.read()
            if ret == False:
                break
            if i >= dlength:
                break
            pbar.update(1)
            j2d = np.zeros([joints2d.shape[0], 2, 25, 3])
            dekr17_to_op_map = [0, 0, 6, 8, 10, 5, 7, 9, 0, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
            op_kpts = joints2d[:, :, dekr17_to_op_map]
            op_kpts[:, :, 1] = (op_kpts[:, :, 2] + op_kpts[:, :, 5]) / 2
            op_kpts[:, :, 8] = (op_kpts[:, :, 9] + op_kpts[:, :, 12]) / 2
            j2d[:, :, :19] = op_kpts[:joints2d.shape[0]]
            j2d_ = j2d[i]
            img_w_kpts2d = plot_skel_cv2(img, j2d_)
            # plot(img_w_kpts2d)
            outp = out_path / Path(f"{seq_n}/{act_name}/{i:04d}.png")
            outp.parent.mkdir(exist_ok=True, parents=True)
            cv2.imwrite(str(outp), img_w_kpts2d)
            imgs.append(img_w_kpts2d)
            i += 1
    make_video(outp)


if __name__ == "__main__":
    # main()
    # plot_skeleton_2d()
    inspect_dekr_detections()
