import os
import glob
import typing

import imageio
import numpy as np
import json

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from body_model import OP_NUM_JOINTS, SMPL_JOINTS
from util.logger import Logger
from geometry.camera import invert_camera

from .tools import read_keypoints, read_mask_path, load_smpl_preds
from .vidproc import preprocess_cameras, preprocess_frames, preprocess_tracks
from .simulation.simu_data import get_simu_data
from util.tensor import get_device, move_to, detach_all, to_torch, to_np


"""
Define data-related constants
"""
DEFAULT_GROUND = np.array([0.0, -1.0, 0.0, -0.5])

# XXX: TEMPORARY CONSTANTS
SHOT_PAD = 0
MIN_SEQ_LEN = 20
MAX_NUM_TRACKS = 12
MIN_TRACK_LEN = 20
MIN_KEYP_CONF = 0.4


def get_dataset_from_cfg(cfg, check_sources=True):
    args = cfg.data
    if not args.use_cams:
        args.sources.cameras = ""

    args.sources = expand_source_paths(args.sources)
    print("DATA SOURCES", args.sources)
    if check_sources:
        check_data_sources(args)
    return MultiPeopleDataset(
        args.sources,
        args.seq,
        tid_spec=args.track_ids,
        shot_idx=args.shot_idx,
        start_idx=int(args.start_idx),
        end_idx=int(args.end_idx),
        split_cameras=args.get("split_cameras", True),
        data_name=args.data_name,
        exp_name=args.exp_name,
        subj_name=args.seq_id,
    )


def expand_source_paths(data_sources):
    return {k: get_data_source(v) for k, v in data_sources.items()}


def get_data_source(source):
    matches = glob.glob(source)
    if len(matches) < 1:
        print(f"{source} does not exist")
        return source  # return anyway for default values
    if len(matches) > 1:
        raise ValueError(f"{source} is not unique")
    return matches[0]


def check_data_sources(args):
    # if args.type == "video" or args.type == "chi3d":
    # ffmpeg
    preprocess_frames(args.sources.images, args.src_path, **args.frame_opts)
    #PHALP
    preprocess_tracks(args.sources.images, args.sources.tracks, args.sources.shots, args.overwrite)
    # SLAM
    preprocess_cameras(args, overwrite=args.get("overwrite_cams", False))


class MultiPeopleDataset(Dataset):
    def __init__(
        self,
        data_sources: typing.Dict,
        seq_name,
        tid_spec="all",
        shot_idx=0,
        start_idx=0,
        end_idx=-1,
        pad_shot=False,
        split_cameras=True,
        data_name=None,
        exp_name=None,
        subj_name=None,
    ):
        self.seq_name = seq_name
        self.data_sources = data_sources
        self.split_cameras = split_cameras
        self.data_name = data_name
        self.exp_name = exp_name
        self.subj_name = subj_name

        # select only images in the desired shot
        img_files, _ = get_shot_img_files(
            self.data_sources["shots"], shot_idx, pad_shot
        )
        end_idx = end_idx if end_idx > 0 else len(img_files)
        self.data_start, self.data_end = start_idx, end_idx
        img_files = img_files[start_idx:end_idx]
        self.img_names = [get_name(f) for f in img_files]
        self.num_imgs = len(self.img_names)

        img_dir = self.data_sources["images"]
        assert os.path.isdir(img_dir)
        self.img_paths = [os.path.join(img_dir, f) for f in img_files]
        img_h, img_w = imageio.imread(self.img_paths[0]).shape[:2]
        self.img_size = img_w, img_h
        print(f"USING TOTAL {self.num_imgs} {img_w}x{img_h} IMGS")

        # find the tracks in the video
        track_root = self.data_sources["tracks"]
        if tid_spec == "all" or tid_spec.startswith("longest"):
            n_tracks = MAX_NUM_TRACKS
            if tid_spec.startswith("longest"):
                n_tracks = int(tid_spec.split("-")[1])
            # get the longest tracks in the selected shot
            track_ids = sorted(os.listdir(track_root))
            track_paths = [
                [f"{track_root}/{tid}/{name}_keypoints.json" for name in self.img_names]
                for tid in track_ids
            ]
            track_lens = [
                len(list(filter(os.path.isfile, paths))) for paths in track_paths
            ]
            track_ids = [
                track_ids[i]
                for i in np.argsort(track_lens)[::-1]
                if track_lens[i] > MIN_TRACK_LEN
            ]
            print("TRACK LENGTHS", track_ids, track_lens)
            track_ids = track_ids[:n_tracks]
        else:
            track_ids = [f"{int(tid):03d}" for tid in tid_spec.split("-")]

        print("TRACK IDS", track_ids)

        self.track_ids = track_ids
        self.n_tracks = len(track_ids)
        self.track_dirs = [os.path.join(track_root, tid) for tid in track_ids]

        # keep a list of frame index masks of whether a track is available in a frame
        sidx = np.inf
        eidx = -1
        self.track_vis_masks = []
        for pred_dir in self.track_dirs:
            # self.img_names
            vis_mask = [True] * len(self.img_names)
            vis_mask = np.array(vis_mask)
            self.track_vis_masks.append(vis_mask)

        eidx = len(self.img_names)
        sidx = 0
        print("START", sidx, "END", eidx)
        self.start_idx = sidx
        self.end_idx = eidx
        self.seq_len = eidx - sidx
        self.seq_intervals = [(sidx, eidx) for _ in track_ids]

        self.sel_img_paths = self.img_paths[sidx:eidx]
        self.sel_img_names = self.img_names[sidx:eidx]

        # used to cache data
        self.data_dict = {}
        self.cam_data = None

    def __len__(self):
        return self.n_tracks

    def load_data(self, interp_input=True):
        if len(self.data_dict) > 0:
            return

        self.load_simulation_data()
        # load camera data
        self.load_camera_data()
        # get data for each track
        data_out = {
            "mask_paths": [],
            "floor_plane": [],
            "joints2d": [],
            "vis_mask": [],
            "track_interval": [],
            "init_body_pose": [],
            "init_root_orient": [],
            "init_trans": [],
        }

        # create batches of sequences
        # each batch is a track for a person
        try:
            simu_data = self.simu_data[self.data_name][self.seq_name]
        except:
            try:
                simu_data = self.simu_data[self.subj_name][self.seq_name]
            except:
                simu_data = self.simu_data[self.subj_name][f"{self.subj_name}_{self.seq_name}"]

        data_out = to_np(move_to(simu_data['obs_data'], 'cpu'))
        # to list only the bacth dim
        for k,v in data_out.items():
            data_out[k] = [c for c in v]

        self.data_dict = data_out

    def __getitem__(self, idx):
        if len(self.data_dict) < 1:
            self.load_data()

        obs_data = dict()

        # 2D keypoints
        joint2d_data = self.data_dict["joints2d"][idx]
        obs_data["joints2d"] = torch.Tensor(joint2d_data)

        # single frame predictions
        obs_data["init_body_pose"] = torch.Tensor(self.data_dict["init_body_pose"][idx])
        obs_data["init_root_orient"] = torch.Tensor(
            self.data_dict["init_root_orient"][idx]
        )
        obs_data["init_trans"] = torch.Tensor(self.data_dict["init_trans"][idx])

        # floor plane
        obs_data["floor_plane"] = torch.Tensor(self.data_dict["floor_plane"][idx])

        # the frames the track is visible in
        obs_data["vis_mask"] = torch.Tensor(self.data_dict["vis_mask"][idx])

        # the frames used in this subsequence
        obs_data["seq_interval"] = torch.Tensor(list(self.seq_intervals[idx])).to(
            torch.int
        )
        # the start and end interval of available keypoints
        obs_data["track_interval"] = torch.Tensor(
            self.data_dict["track_interval"][idx]
        ).int()

        obs_data["track_id"] = int(self.track_ids[idx])
        obs_data["seq_name"] = self.seq_name

        debug = False
        if debug:
            from utils.smpl import smpl_to_verts
            from utils.misc import save_mesh
            root = "/home/nugrinovic/code/CVPR_2024/slahmr_release/slahmr/"
            op = f"{root}/inspect_out/init_simu_poses/"
            
            root =  obs_data["init_root_orient"]
            body_pose =  obs_data["init_body_pose"].reshape([-1, 63])
            trans =  obs_data["init_trans"]
            betas = self.data_dict["init_betas"][idx]
            B = len(root)
            pose_aa = torch.cat([root, body_pose, torch.zeros(B, 6).to(root)], dim=1)  # (1, 72)

            verts, faces = smpl_to_verts(pose_aa, trans, betas=betas, return_joints=False)
            verts = verts[0, ::5]
            save_mesh(verts, faces, f"{op}/poses_simu.ply")


        return obs_data

    def load_camera_data(self):
        cam_dir = self.data_sources["cameras"]
        data_interval = 0, -1
        if self.split_cameras:
            data_interval = self.data_start, self.data_end
        track_interval = self.start_idx, self.end_idx
        self.cam_data = CameraData(
            cam_dir, self.seq_len, self.img_size, data_interval, track_interval,
            simu_data=self.simu_data,
            data_name=self.data_name,
            subj_name=self.subj_name,
            seq_name=self.seq_name,
        )

    def load_simulation_data(self):
        data = get_simu_data(self.data_name, self.exp_name)
        self.simu_data = data

    def get_camera_data(self):
        if self.cam_data is None:
            raise ValueError
        return self.cam_data.as_dict()


class CameraData(object):
    def __init__(
        self, cam_dir, seq_len, img_size, data_interval=[0, -1], track_interval=[0, -1],
            simu_data=None,
            data_name=None,
            subj_name=None,
            seq_name=None,
    ):
        assert simu_data is not None, "simu_data is None!!"
        assert data_name is not None, "data_name is None!!"
        assert subj_name is not None, "subj_name is None!!"
        assert seq_name is not None, "seq_name is None!!"

        self.img_size = img_size
        self.cam_dir = cam_dir
        
        self.simu_data = simu_data
        self.data_name = data_name
        self.subj_name = subj_name
        self.seq_name = seq_name

        # inclusive exclusive
        data_start, data_end = data_interval
        if data_end < 0:
            data_end += seq_len + 1
        data_len = data_end - data_start

        # start and end indices are with respect to the data interval
        sidx, eidx = track_interval
        if eidx < 0:
            eidx += data_len + 1
        self.sidx, self.eidx = sidx + data_start, eidx + data_start
        self.seq_len = self.eidx - self.sidx

        self.load_data()

    def load_data(self):
        # camera info
        img_w, img_h = self.img_size
        # fpath = os.path.join(self.cam_dir, "cameras.npz")
        # Logger.log(f"WARNING: {fpath} does not exist, using static cameras...")
        # get cams from simu data
        try:
            cam_data = self.simu_data[self.data_name][self.seq_name]['cam_data']
        except:
            try:
                cam_data = self.simu_data[self.subj_name][self.seq_name]['cam_data']
            except:
                cam_data = self.simu_data[self.subj_name][f"{self.subj_name}_{self.seq_name}"]['cam_data']

        self.intrins = cam_data['intrins'].cpu()
        self.cam_R = cam_data['cam_R'].cpu()
        self.cam_t = cam_data['cam_t'].cpu()
        self.is_static = True

        Logger.log(f"Images have {img_w}x{img_h}, intrins {self.intrins[0]}")
        print("CAMERA DATA", self.cam_R.shape, self.cam_t.shape, self.intrins[0])

    def world2cam(self):
        return self.cam_R, self.cam_t

    def cam2world(self):
        R = self.cam_R.transpose(-1, -2)
        t = -torch.einsum("bij,bj->bi", R, self.cam_t)
        return R, t

    def as_dict(self):
        return {
            "cam_R": self.cam_R,  # (T, 3, 3)
            "cam_t": self.cam_t,  # (T, 3)
            "intrins": self.intrins,  # (T, 4)
            "static": self.is_static,  # bool
        }


def get_ternary_mask(vis_mask):
    # get the track start and end idcs relative to the filtered interval
    vis_mask = torch.as_tensor(vis_mask)
    vis_idcs = torch.where(vis_mask)[0]
    track_s, track_e = min(vis_idcs), max(vis_idcs) + 1
    # -1 = track out of scene, 0 = occlusion, 1 = visible
    vis_mask = vis_mask.float()
    vis_mask[:track_s] = -1
    vis_mask[track_e:] = -1
    return vis_mask


def get_shot_img_files(shots_path, shot_idx, shot_pad=SHOT_PAD):
    assert os.path.isfile(shots_path), f"prob. wrong path {shots_path}"
    with open(shots_path, "r") as f:
        shots_dict = json.load(f)
    img_names = sorted(shots_dict.keys())
    N = len(img_names)
    shot_mask = np.array([shots_dict[x] == shot_idx for x in img_names])

    idcs = np.where(shot_mask)[0]
    if shot_pad > 0:  # drop the frames before/after shot change
        if min(idcs) > 0:
            idcs = idcs[shot_pad:]
        if len(idcs) > 0 and max(idcs) < N - 1:
            idcs = idcs[:-shot_pad]
        if len(idcs) < MIN_SEQ_LEN:
            raise ValueError("shot is too short for optimization")

        shot_mask = np.zeros(N, dtype=bool)
        shot_mask[idcs] = 1
    sel_paths = [img_names[i] for i in idcs]
    print(f"FOUND {len(idcs)}/{len(shots_dict)} FRAMES FOR SHOT {shot_idx}")
    return sel_paths, idcs


def load_cameras_npz(camera_path):
    assert os.path.splitext(camera_path)[-1] == ".npz"

    cam_data = np.load(camera_path)
    height, width, focal = (
        int(cam_data["height"]),
        int(cam_data["width"]),
        float(cam_data["focal"]),
    )

    w2c = torch.from_numpy(cam_data["w2c"])  # (N, 4, 4)
    cam_R = w2c[:, :3, :3]  # (N, 3, 3)
    cam_t = w2c[:, :3, 3]  # (N, 3)
    N = len(w2c)

    if "intrins" in cam_data:
        intrins = torch.from_numpy(cam_data["intrins"].astype(np.float32))
    else:
        intrins = torch.tensor([focal, focal, width / 2, height / 2])[None].repeat(N, 1)

    print(f"Loaded {N} cameras")
    return cam_R, cam_t, intrins, width, height


def is_image(x):
    return (x.endswith(".png") or x.endswith(".jpg")) and not x.startswith(".")


def get_name(x):
    return os.path.splitext(os.path.basename(x))[0]


def split_name(x, suffix):
    return os.path.basename(x).split(suffix)[0]


def get_names_in_dir(d, suffix):
    files = [split_name(x, suffix) for x in glob.glob(f"{d}/*{suffix}")]
    return sorted(files)


def batch_join(parent, names, suffix=""):
    return [os.path.join(parent, f"{n}{suffix}") for n in names]
