import numpy as np
from pathlib import Path
from tqdm import tqdm
from utils.misc import write_pickle
from collections import defaultdict
from utils.smpl_transforms import rot_and_correct_smplx_offset_full
from pyquaternion import Quaternion as Q


def get_sequence_data(smpl_dir, cam_dict, action, meta_list, rot_180z=False):
    Tx90 = Q(axis=[1, 0, 0], angle=np.pi / 2).transformation_matrix.astype(np.float32)
    Tz180 = Q(axis=[0, 0, 1], angle=np.pi).transformation_matrix.astype(np.float32)

    data_dict = defaultdict(list)
    files = sorted(smpl_dir.glob('*.npz'))
    for file in files:
        data = np.load(file)
        body_pose = data['body_pose']
        glob_ori = data['global_orient']
        transl = data['transl']
        betas = data['betas']

        pose_aa = np.concatenate([glob_ori, body_pose], axis=-1)
        if rot_180z:
            Tr = Tz180[None] @ Tx90[None]
        else:
            Tr = Tx90[None]
        pose_aa_r, trans_r = rot_and_correct_smplx_offset_full(pose_aa, transl, betas, Tr, get_verts=False)
        glob_ori = pose_aa_r[:, :3]
        transl = trans_r.squeeze()
        data_dict['root_orient'].append(glob_ori)
        data_dict['pose_body'].append(body_pose)
        data_dict['trans'].append(transl)
        data_dict['betas'].append(betas)

    data_dict = {k: np.stack(v) for k, v in data_dict.items()}
    return data_dict


def parse_cam_data(cam_data):
    # this is K (B, 3, 3)
    intrinsics = cam_data['intrinsics']
    # this is R|t (B, 3, 4)
    extrinsics = cam_data['extrinsics']
    cam_id = 0  # 0 for images in folder 4
    R = extrinsics[cam_id, :3, :3]
    T = extrinsics[cam_id, :3, 3][None]
    # this is for visu only
    focal = [intrinsics[cam_id, 0, 0], intrinsics[cam_id, 1, 1]]
    center = [intrinsics[cam_id, 0, 2], intrinsics[cam_id, 1, 2]]
    K = {'f': focal, 'c': center}
    cam_dict = {'R': R, 'T': T, 'K': K}
    return cam_dict


def parse_meta(meta_data):
    start = meta_data['start'].item()
    end = meta_data['end'].item()
    meta_list = [start, end]
    return meta_list


def main(args, debug=False):
    data_name = "hi4d"
    args.data_name = data_name
    # place the path to the dataset dir HERE
    DATA_ROOT = Path("/path/to/hi4d/dataset")
    imgs_root = DATA_ROOT / "data"
    out_dir = Path("./data/hi4d")

    subjects = sorted(imgs_root.glob('*'))
    # filter "video" folder
    subjects = [subj for subj in subjects if not subj.stem.startswith('video')]
    if args.rot_180z:
        print("**WARNING: will rotate GT by 180z to match EmbPose!!**")

    data_dict_p1, data_dict_p2 = {}, {}
    act_cnt = 0
    for n, subj in enumerate(tqdm(subjects)):
        actions = sorted(subj.glob('*'))
        if "pair" in actions[0].name:
            # if did not get the correct folders, should go one level deeper
            actions = sorted(subj.glob('*/*'))
        for action in actions:
            sub = subj.stem
            act = action.stem
            name = f"{sub}_{act}"

            if args.filter_seq is not None:
                if name != args.filter_seq:
                    continue

            act_cnt += 1
            smpl_dir = action / 'smpl'
            cam_path = action / 'cameras' /'rgb_cameras.npz'
            meta_path = action / 'meta.npz'
            try:
                cam_data = np.load(cam_path)
            except:
                print(f'No cam data for {cam_path}')
                continue
            cam_dict = parse_cam_data(cam_data)
            try:
                meta_data = np.load(meta_path)
                meta_list = parse_meta(meta_data)
            except:
                meta_list = []

            data_dict = get_sequence_data(smpl_dir, cam_dict, action, meta_list, rot_180z=args.rot_180z)
            emb_data_p1 = {k: v[:, 0] for k, v in data_dict.items()}
            emb_data_p2 = {k: v[:, 1] for k, v in data_dict.items()}
            # as it is right now, this cam is actually world2cam, todo change
            emb_data_p1["cam2world"] = cam_dict
            emb_data_p2["cam2world"] = cam_dict

            data_dict_p1[f"{sub}_{act}"] = emb_data_p1
            data_dict_p2[f"{sub}_{act}"] = emb_data_p2

    out_path = out_dir / Path(f'{data_name}_embodied_cam2w_p1.pkl')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_pickle(data_dict_p1, out_path)

    out_path = out_dir / Path(f'{data_name}_embodied_cam2w_p2.pkl')
    write_pickle(data_dict_p2, out_path)

    print(f"saved to {out_path}")
    print(f"Total actions: {act_cnt} ")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', type=str, default=None)
    parser.add_argument("--filter_seq", type=str, default=None)
    parser.add_argument('--rot_180z', type=int, default=1, choices=[0, 1])
    args = parser.parse_args()

    main(args)
