from functools import partial
from math import pi
from pathlib import Path
import numpy as np
import roma
import smplx
import torch
import trimesh
from pyquaternion import Quaternion as Q
from tqdm import tqdm
from utils.body_model import pose_to_vertices as pose_to_vertices_
from utils.chi3d_util import project_3d_to_2d_simple
from utils.chi3d_util import read_cam_params
from utils.misc import read_json
from utils.misc import write_pickle
import os
from multiphys.data_process.rigid_transform_3d import get_cam2world


Trx = Q(axis=[1, 0, 0], angle=-pi / 2.).transformation_matrix.astype(np.float32)
Trx = torch.from_numpy(Trx[None, :3, :3])# .cuda()

SEQS_train = ['s02', 's03', 's04']
SEQS_test = ['s01', 's05']
ACTS = ['Kick', 'Push', 'Grab', 'Posing', 'HoldingHands', 'Handshake', 'Hit', 'Hug']
data_keys = ['transl', 'global_orient', 'body_pose', 'betas', 'left_hand_pose', 'right_hand_pose', 'jaw_pose',
             'leye_pose', 'reye_pose', 'expression']

CAMERAS = [
    "50591643",
    "58860488",
    "60457274",
    "65906101",
]

embodied_keys = ['transl', 'global_orient', 'body_pose', 'betas']

keys_name_map = {
    "transl": "trans",
    "global_orient": "root_orient",
    "body_pose": "pose_body",
    "betas": "betas",
}

chi3d_to_dekr_map = [11, 14, 12, 15, 13, 16, 1, 4, 2, 5, 3, 6, 10, 8]
chi3d_to_op_map = [9, 8, 14, 15, 16, 11, 12, 13, 0, 4, 5, 6, 1, 2, 3]

local_bm = smplx.create("data", 'smpl', use_pca=False, batch_size=1)#.cuda()
pose_to_vertices = partial(pose_to_vertices_, pose_type="smpl", alpha=1, bm=local_bm)


def parse_one_person(data, kpts3d_path, file, idx=0):
    data_p1 = {}
    for k, v in data.items():
        v = torch.tensor(v[idx]).float()
        if k != 'transl' and k != 'betas' and k != 'expression':
            v = roma.rotmat_to_rotvec(v).reshape([v.shape[0], -1])
        data_p1[k] = v

    kpt3d_ = read_json(kpts3d_path / Path(f'{file.stem}.json'))
    kpt3d = np.array(kpt3d_["joints3d_25"])
    kpt3d = torch.from_numpy(kpt3d).float()
    data_p1['joints3d'] = kpt3d[idx]

    return data_p1


def chi3d_to_emb(data_p1, cam_path, file, downsample=False):
    emb_data_p = {}
    for k, v in data_p1.items():
        if k in embodied_keys:
            new_k = keys_name_map[k]
            emb_data_p[new_k] = v

    for cam in CAMERAS:
        cam_params = read_json(cam_path / Path(f'{cam}/{file.stem}.json'))
        ext = cam_params["extrinsics"]
        R = ext["R"]
        T = ext["T"][0]
        K = cam_params['intrinsics_wo_distortion']
        cam_dict = {'R': R, 'T': T, 'K': K}
        emb_data_p[f'cam_{cam}'] = cam_dict


    # focal_lenght = 1000
    # img_size = (900, 900)
    cam = "50591643"
    cam_params = read_cam_params(cam_path / Path(f'{cam}/{file.stem}.json'))
    T = cam_params['extrinsics']['T'] # (1, 3)
    R = cam_params['extrinsics']['R'] # (3, 3)
    kpts3d_p1 = np.array(data_p1["joints3d"]) # (219, 25, 3)
    j3d_in_camera = np.matmul(kpts3d_p1 - T[None], R[None].transpose(0, 2, 1)) # (219, 25, 3)
    kpts2d = project_3d_to_2d_simple(j3d_in_camera, cam_params['intrinsics_wo_distortion'])
    op_kpts = kpts2d[:, chi3d_to_op_map]
    emb_kpts = np.zeros_like(kpts2d)
    # nose, same as OP nose
    emb_kpts[:, 0] = kpts2d[:, 9] # take only 1 jt from OP
    emb_kpts[:, :15] = op_kpts # dekr are 14 jts

    # this is in the emb format, add confidences
    conf = np.zeros_like(emb_kpts[0, :, 0])
    conf[:15] = 1
    nconf = np.broadcast_to(conf, (emb_kpts.shape[0], conf.shape[0])) # (2, 3)
    emb_kpts_w_confs = np.concatenate([emb_kpts, nconf[..., None]], axis=-1)

    emb_data_p['joints2d'] = emb_kpts_w_confs

    rot = emb_data_p["root_orient"]
    poseb = emb_data_p["pose_body"]
    trans = emb_data_p["trans"]
    pose = torch.cat([rot, poseb], axis=1)
    poses_72 = torch.zeros([1, 72])
    pose_in = pose[0, None]
    poses_72[:, :66] = pose_in
    poses_75 = torch.cat([poses_72, trans[0, None]], 1)
    verts = pose_to_vertices(poses_75[None])
    mesh = trimesh.Trimesh(vertices=verts[0, 0].detach().cpu().numpy(), faces=local_bm.faces)
    bbox = mesh.bounding_box.bounds
    min_xyz = bbox[0]
    trans_floor_z = trans[:, 2] - min_xyz[None, 2]
    new_trans = torch.cat([trans[:, :2], trans_floor_z[:, None]], 1)
    emb_data_p["trans"] = new_trans

    cam2wold = get_cam2world(emb_data_p)
    emb_data_p["cam2world"] = cam2wold

    if downsample:
        emb_data_p['joints2d'] = emb_data_p['joints2d'][::2]
        emb_data_p['betas'] = emb_data_p['betas'][::2]
        emb_data_p['trans'] = emb_data_p['trans'][::2]
        emb_data_p['root_orient'] = emb_data_p['root_orient'][::2]
        emb_data_p['pose_body'] = emb_data_p['pose_body'][::2]

    return emb_data_p


def main(split='train'):
    dataname = 'chi3d'
    base_dir = '/path/to/chi3d/dataset'
    out_dir = Path("./data/chi3d/")

    SEQS = SEQS_train if split == 'train' else SEQS_test
    data_dir = os.path.join(base_dir, '{0}')
    data_dir = data_dir.format(split, '{}')
    data_dir = Path(data_dir)
    cam_path = os.path.join(base_dir, '{0}/{1}/camera_parameters')
    kpts3d_path = os.path.join(base_dir, '{0}/{1}/joints3d_25')

    for seq in SEQS:
        print(f"Doing {seq}")
        cam_path_p = Path(cam_path.format(split, seq, '{}'))
        kpts3d_path = Path(str(kpts3d_path).format(split, seq, '{}'))
        smpl_dir = data_dir / Path(f'{seq}/smplx')
        files = sorted(list(smpl_dir.glob('*.json')))
        data_dict_p1 = {}
        data_dict_p2 = {}
        for n, file in enumerate(tqdm(files)):
            name = file.stem.replace(' ', '_')
            data = read_json(file)
            for k, v in data.items():
                data[k] = np.array(v)
            try:
                data_p1 = parse_one_person(data, kpts3d_path, file, idx=0)
                data_p2 = parse_one_person(data, kpts3d_path, file, idx=1)
            except:
                print(f"Error reading {file}")
                continue
            emb_data_p1 = chi3d_to_emb(data_p1, cam_path_p, file)
            emb_data_p2 = chi3d_to_emb(data_p2, cam_path_p, file)

            data_dict_p1[f"{seq}_{name}"] = emb_data_p1
            data_dict_p2[f"{seq}_{name}"] = emb_data_p2

        out_path = out_dir / Path(f'{dataname}_{seq}_embodied_cam2w_p1.pkl')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_pickle(data_dict_p1, out_path)
        out_path = out_dir / Path(f'{dataname}_{seq}_embodied_cam2w_p2.pkl')
        write_pickle(data_dict_p2, out_path)
        print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
