from utils.visu_tools import extract_video_frames
from utils.visu_tools import make_video
import numpy as np
from pathlib import Path
from utils.misc import read_image_PIL
from utils.misc import plot_skel_cv2
import os.path as osp
from tqdm import tqdm
from utils.misc import save_img
import cv2
import joblib

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def main(args):
    ROOT = "/sailhome/nugrinov/code/CVPR_2024/EmbodiedPose"
    path = f"{ROOT}/sample_data/hi4d/thirdeye_clip_hi4d_embodied_cam2w_p1_debug_floor_phalp_initSla.pkl"
    path2 = path.replace("_p1_", "_p2_")
    data = joblib.load(path)
    data2 = joblib.load(path2)

    output_dir = f"{ROOT}/inspect_out/hi4d/matching"

    for take_key in data:
        jts = data[take_key]["joints2d"]
        jts2 = data2[take_key]["joints2d"]
        joints2d_all = np.stack([jts, jts2], axis=0)
        j2_len = len(jts)

        pass
        print("EXTRACTING original video frames ")
        img_files = extract_video_frames(args.data_name, take_key, sub_name=None)

        out_joint_imgs = osp.join(output_dir, f"kpts_2d")

        Path(out_joint_imgs).mkdir(parents=True, exist_ok=True)
        print('Saving joint images...')
        for n, file in enumerate(tqdm(img_files[:j2_len])):
            img = read_image_PIL(file)
            j2d = joints2d_all[n]

            # if img is too big, resize it, otherwise it kills the process
            h, w, _ = img.shape
            if h == 2048 and w==2048:
                img = cv2.resize(img, (h//2, h//2))
                j2d[..., :2] = j2d[..., :2] / 2

            # this plot_skel_cv2 works for multi-person, just pass an array with batch dim = num people
            img_w_kpts2d = plot_skel_cv2(img, j2d)
            outp = Path(f"{out_joint_imgs}/{n:06d}.png")
            outp.parent.mkdir(parents=True, exist_ok=True)
            save_img(outp, img_w_kpts2d)

        make_video(outp)  # img here are always png


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default="hi4d", choices=["chi3d", 'hi4d'])
    args = parser.parse_args()

    main(args)
