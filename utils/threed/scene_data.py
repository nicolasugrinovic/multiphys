import os
import torch
import trimesh
import numpy as np
from pathlib import Path
from utils.misc import read_pickle
from utils.misc import read_json
from utils.misc import write_json
from utils.misc import read_npy
from argparse import ArgumentParser
from tqdm import tqdm
from utils.misc import save_trimesh
from utils.misc import save_mesh
from utils.misc import save_pointcloud
# from pyquaternion import Quaternion as Q
from utils.pyquaternion import Quaternion as Q
from math import pi
from trimesh import creation, transformations

# Transformation for 90 degree rotations around the x axis
T1 = Q(axis=[1, 0, 0], angle=pi / 2.).transformation_matrix.astype(np.float32)

rot_matrix = transformations.rotation_matrix(angle=pi / 2., direction=[1, 0, 0], point=[0, 0, 0]) #point=center


def generate_scene_translations(scene_verts, static_scene, cams_dir, scene_name):
    ys = scene_verts[:, 1]
    ys_sorted = np.sort(ys)
    max_ys = ys_sorted[-1000:]
    max_y_mean = np.mean(max_ys)
    scene_verts[:, 1] -= max_y_mean
    static_scene.vertices = scene_verts
    ys_out_path = cams_dir / Path(scene_name + '_transl.json')
    data = {'max_y_mean': np.asarray(max_y_mean).item()}
    write_json(ys_out_path, data)

def get_prox_scenes_data(args, device, scene_norm=True, hi_res=False, generate_scene_transl_info=False):
    root_dir = './data/prox_with_lemo'
    scene_dir = Path(root_dir) / Path('scenes_downsampled')
    # scene_dir = Path(root_dir) / Path('scenes_floors')
    if hi_res:
        scene_dir = Path(root_dir) / Path('scenes')
    cams_dir = Path(root_dir) / Path('cam2world')
    sdf_dir = Path(root_dir) / Path('sdf')

    # self.meta_dir_t = os.path.join(self.args.train_data_dir, f'meta.pkl')
    # meta_dir_v = os.path.join(args.val_data_dir, f'meta.pkl')
    meta_v = None
    if args.do_scenes_eval:
        # meta_v = np.array(read_pickle(meta_dir_v))
        meta_v = None
        print('********WARNING: meta_v is None when generating scenes data for PROX val!!!!!!!')

    scene_paths = list(Path(scene_dir).glob('*.ply'))

    if args.dummy:
        static_scenes = {}
        for spath in scene_paths:
            scene_name = spath.stem
            static_scenes[scene_name] = {}
            static_scenes[scene_name]['scene_pc'] = torch.zeros([50000, 3]).float()
            static_scenes[scene_name]['scene_rgb'] = torch.zeros([50000, 3]).float()
            static_scenes[scene_name]['camera_extrinsic'] = torch.zeros([4, 4]).float()
            static_scenes[scene_name]['sdf'] = torch.zeros([256, 256, 256]).float()
            static_scenes[scene_name]['grid_min'] = torch.zeros([3, ]).float()
            static_scenes[scene_name]['grid_max'] = torch.zeros([3, ]).float()
            static_scenes[scene_name]['scene_y'] = torch.zeros([1]).float()
    else:
        scene_lst = []
        faces_lst = []
        rgb_lst = []
        names = []
        cameras_extrinsics = []
        ytransl_list = []

        for spath in tqdm(scene_paths):
            scene_name = spath.stem
            names.append(scene_name)
            static_scene = trimesh.load(spath, process=False)
            n_verts = static_scene.vertices.shape[0]
            if not hi_res:
                assert n_verts == 50000, f"Scene mesh has {n_verts} vertices, expected 50000, check scene mseshes or trimesh.load"
            cam_ext_path = cams_dir / Path(scene_name + '.json')
            cam_ext = read_json(cam_ext_path)
            scene_rgb = static_scene.visual.vertex_colors

            ys_path = cams_dir / Path(scene_name + '_transl.json')
            ys = read_json(ys_path)
            max_y_mean = ys['max_y_mean']

            if scene_norm:
                static_scene.apply_transform(rot_matrix)
                scene_verts = static_scene.vertices
                scene_verts[:, 1] -= max_y_mean
                static_scene.vertices = scene_verts

                # to generate the translation info to normalize the scene. This should never be called again actually
                # used to generate this files and should not change
                if generate_scene_transl_info:
                    generate_scene_translations(scene_verts, static_scene, cams_dir, scene_name)

                if 0:
                    path = Path('inspect_out/meshes/prox_scenes/norm_rot90x_transl')
                    if hi_res:
                        path = Path('inspect_out/meshes/prox_scenes/norm_rot90x_transl_hi_res')
                    path.mkdir(parents=True, exist_ok=True)
                    static_scene.export(path / Path(scene_name + '.ply'));
                    print(f'Exported scene {scene_name} to {path}')

                    static_scene.vertices = scene_verts * np.array([[-1, 1, 1]])
                    path = Path('inspect_out/meshes/prox_scenes/norm_rot90x_transl_flipped')
                    path.mkdir(parents=True, exist_ok=True)
                    static_scene.export(path / Path(scene_name + '_flipped.ply'));
                    # flip over z axis, for some re,ason, bug in the SDF loss

                    static_scene.vertices = scene_verts * np.array([[1, 1, -1]])
                    path = Path('inspect_out/meshes/prox_scenes/norm_rot90x_transl_flipped_z')
                    path.mkdir(parents=True, exist_ok=True)
                    static_scene.export(path / Path(scene_name + '_flipped_z.ply'));
                    continue
            else:
                # this transformation takes the scene into camera space (same space as the motions)
                cam_ = np.linalg.inv(cam_ext)
                static_scene.apply_transform(cam_)
                # save_pointcloud(static_scene.vertices[:, :3], f'inspect_out/meshes/prox_not_norm/{scene_name}.ply')
                # static_scene.export(path/ Path(scene_name + '_inv_transf.ply'));

            rgb_lst.append(torch.tensor(scene_rgb).float())
            ytransl_list.append(torch.tensor([max_y_mean]).float())
            # aaaa = torch.tensor([max_y_mean]).float()
            # aaaa.shape[0]
            scene_lst.append(torch.tensor(static_scene.vertices).float())
            faces_lst.append(torch.tensor(static_scene.faces).float())
            cameras_extrinsics.append(torch.tensor(cam_ext).float())

        assert len(scene_lst) == 12, f'Number of static scenes should be 12, but got {len(scene_lst)}'

        static_scenes = {}
        # get 3D pc data
        for n, scene_pc in enumerate(scene_lst):
            static_scenes[names[n]] = {}
            static_scenes[names[n]]['scene_pc'] = scene_pc
            static_scenes[names[n]]['scene_rgb'] = rgb_lst[n]
            static_scenes[names[n]]['scene_y'] = ytransl_list[n]
            static_scenes[names[n]]['camera_extrinsic'] = cameras_extrinsics[n]

        # get the SDF data
        sdf_paths = list(Path(sdf_dir).glob('*_sdf.npy'))
        cnt = 0
        print('PROX: reading SDFs...')
        for spath in sdf_paths:
            scene_name = spath.stem.split('_')[0]
            sdf_data = read_json(str(spath).replace('_sdf.npy', '.json'))
            grid_dim = sdf_data['dim']
            grid_min = torch.tensor(np.array(sdf_data['min']), dtype=torch.float32)
            grid_max = torch.tensor(np.array(sdf_data['max']), dtype=torch.float32)
            sdf = np.load(spath).reshape(grid_dim, grid_dim, grid_dim)
            static_scenes[scene_name]['sdf'] = torch.tensor(sdf).float()
            static_scenes[scene_name]['grid_min'] = grid_min
            static_scenes[scene_name]['grid_max'] = grid_max
            # self.static_scenes[scene_name] += (sdf, grid_min, grid_max)
            cnt += 1
        assert cnt == 12, f'Number of sdf scenes should be 12, but got {cnt}'

    # to cuda once
    print('Moving prox scenes to cuda...')
    for k in static_scenes:
        static_scenes[k]['scene_pc'] = static_scenes[k]['scene_pc'].to(device)
        static_scenes[k]['scene_rgb'] = static_scenes[k]['scene_rgb'].to(device)
        static_scenes[k]['scene_y'] = static_scenes[k]['scene_y'].to(device)
        static_scenes[k]['camera_extrinsic'] = static_scenes[k]['camera_extrinsic'].to(device)
        static_scenes[k]['sdf'] = static_scenes[k]['sdf'].to(device)
        static_scenes[k]['grid_min'] = static_scenes[k]['grid_min'].to(device)
        static_scenes[k]['grid_max'] = static_scenes[k]['grid_max'].to(device)

    return static_scenes, meta_v


def get_prox_cams():
    cams_dir = Path('./data/prox_with_lemo/cam2world')
    cam_ext = read_json(cams_dir / Path('MPH112' + '.json'))
    ys_path = cams_dir / Path('MPH112' + '_transl.json')
    ys = read_json(ys_path)
    max_y_mean = ys['max_y_mean']
    return cam_ext, max_y_mean


def get_humanise_scenes_data(args, device, is_val, scene_norm=True):
    scene_dir = 'data/scannet/downsampled_scans'
    scene_paths = sorted(list(Path(scene_dir).glob('*.npy')))
    sdf_dir = Path('data/scannet/sdf/res_064')

    # get cams from a manually chosen prox scene, beware of this!
    cam_ext, max_y_mean = get_prox_cams()

    # T1 = Q(axis=[0, 0, 1], angle=pi/2.).transformation_matrix.astype(np.float32)
    T2 = Q(axis=[-1, 0, 0], angle=pi/2.).transformation_matrix.astype(np.float32)
    T = T2 #@ T1

    if is_val: # only read metrics scenes
        scene_paths = [c for c in scene_paths if int(c.stem.split('_')[0][5:]) >= 600]
    scene_lst, names = [], []

    if args.dummy:
        static_scenes = {}
        for spath in tqdm(scene_paths):
            scene_name = spath.stem
            static_scenes[scene_name] = {}
            static_scenes[scene_name]['scene_pc'] = torch.zeros([50000, 3]).float()
            static_scenes[scene_name]['scene_rgb'] = torch.zeros([50000, 3]).float()
            static_scenes[scene_name]['camera_extrinsic'] = torch.zeros([2]).float()
            # sdf with 256x256x256 gives CUDA out of memory error
            static_scenes[scene_name]['sdf'] = torch.zeros([2]).float()
            static_scenes[scene_name]['grid_min'] = torch.zeros([3]).float()
            static_scenes[scene_name]['grid_max'] = torch.zeros([3]).float()
            static_scenes[scene_name]['scene_y'] = torch.zeros([2]).float()
            static_scenes[scene_name]['rgb'] = torch.zeros([3, ]).float()
            static_scenes[scene_name]['normals'] = torch.zeros([3, ]).float()
            static_scenes[scene_name]['ins_seg'] = torch.zeros([1, ]).float()
            static_scenes[scene_name]['sem_seg'] = torch.zeros([1, ]).float()
    else:
        print('Loading Humanise scenes...')
        if args.debug:
            scene_paths_val = [c for c in scene_paths if int(c.stem.split('_')[0][5:]) >= 600]
            scene_paths = scene_paths[:20]
            scene_paths_ = scene_paths_val[:20]
            scene_paths += scene_paths_

        for spath in tqdm(scene_paths):
            scene_name = spath.stem
            names.append(scene_name)
            # scene_file = f'{spath}/{scene_name}_vh_clean_2.ply'
            # static_scene = trimesh.load(scene_file, process=False)
            scene = np.load(str(spath), allow_pickle=True) # these should be already transformed
            # save_pointcloud(scene[:, :3], f'inspect_out/meshes/scannet_posegpt/{scene_name}.ply')
            if not scene_norm:
                # read any prox ext cam to take scenes to s reasonable 3d space for pre-trained PointNet
                xyz_rot = (T[:3, :3] @ scene[:, :3].T).T
                xyz_rot -= np.array([0, 0, max_y_mean])[None]
                # save_pointcloud(xyz_rot[:, :3], f'inspect_out/meshes/scannet_to_proxcam/{scene_name}_t.ply')
                static_scene = trimesh.PointCloud(xyz_rot)
                cam_ = np.linalg.inv(cam_ext)
                static_scene.apply_transform(cam_)
                # save_pointcloud(static_scene[:, :3], f'inspect_out/meshes/scannet_to_proxcam/{scene_name}_rot.ply')
                scene_tf = np.asarray(static_scene.vertices[:, :3])
                scene[:, :3] = scene_tf

            scene_lst.append(torch.tensor(scene).float())

        static_scenes = {}
        print('Creating scenes dict...')
        for n, scene in enumerate(scene_lst):
            static_scenes[names[n]] = {}
            static_scenes[names[n]]['scene_pc'] = scene[:, :3]
            static_scenes[names[n]]['scene_rgb'] = scene[:,3:6]
            static_scenes[names[n]]['normals'] = scene[:,6:9]
            static_scenes[names[n]]['ins_seg'] = scene[:,9]
            static_scenes[names[n]]['sem_seg'] = scene[:,10]
            static_scenes[names[n]]['camera_extrinsic'] = torch.zeros([2]).float()
            static_scenes[names[n]]['sdf'] = torch.zeros([2]).float()
            static_scenes[names[n]]['grid_min'] = torch.zeros([2]).float()
            static_scenes[names[n]]['grid_max'] = torch.zeros([2]).float()
            static_scenes[names[n]]['scene_y'] = torch.zeros([2]).float()

    # get the SDF data
    exist_paths = [c.stem for c in scene_paths]
    sdf_paths = list(Path(sdf_dir).glob('*_sdf.npy'))
    sdf_paths = sorted([c for c in sdf_paths if c.stem.split('_sdf')[0] in exist_paths])
    # in case we expand to train scenes
    # if is_val: # for now there is only sdf for val scenes
    sdf_paths = [c for c in sdf_paths if int(c.name.split('_')[0][5:]) >= 600]
    cnt = 0
    print('Humanise: reading SDFs...')
    for spath in sdf_paths:
        scene_name = spath.stem.split('_sdf')[0]
        sdf_data = read_json(str(spath).replace('_sdf.npy', '.json'))
        grid_dim = sdf_data['dim']
        grid_min = torch.tensor(np.array(sdf_data['min']), dtype=torch.float32)
        grid_max = torch.tensor(np.array(sdf_data['max']), dtype=torch.float32)
        sdf = np.load(str(spath)).reshape(grid_dim, grid_dim, grid_dim)
        static_scenes[scene_name]['sdf'] = torch.tensor(sdf).float()
        static_scenes[scene_name]['grid_min'] = grid_min
        static_scenes[scene_name]['grid_max'] = grid_max
        # self.static_scenes[scene_name] += (sdf, grid_min, grid_max)
        cnt += 1
    if not args.debug:
        assert cnt == 224, f'Number of VAL sdf scenes in Humanise should be 224, but got {cnt}'

    # to cuda once
    print('Moving scenes to cuda...')
    for k in static_scenes:
        static_scenes[k]['scene_pc'] = static_scenes[k]['scene_pc'].to(device)
        static_scenes[k]['scene_rgb'] = static_scenes[k]['scene_rgb'].to(device)
        static_scenes[k]['normals'] = static_scenes[k]['normals'].to(device)
        static_scenes[k]['sem_seg'] = static_scenes[k]['sem_seg'].to(device)
        # static_scenes[k]['camera_extrinsic'] = static_scenes[k]['camera_extrinsic'].to(device)
        # static_scenes[k]['sdf'] = static_scenes[k]['sdf'].to(device)
        # static_scenes[k]['grid_min'] = static_scenes[k]['grid_min'].to(device)
        # static_scenes[k]['grid_max'] = static_scenes[k]['grid_max'].to(device)

    meta_v = None
    return static_scenes, meta_v

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--do_scenes_eval", type=int, default=0, choices=[0, 1])
    parser.add_argument("--val_data_dir", type=str, default='data/prox_with_lemo/prox/val_47/abs_pose_seqLen64_fps30_overlap0_minSeqLen16')
    parser.add_argument("--dummy", type=int, default=0, choices=[0, 1])
    parser.add_argument("--debug", type=int, default=0, choices=[0, 1])
    args = parser.parse_args()

    scene_opts = {
        'scene_norm': True,
        'hi_res': False,
    }

    # get_humanise_scenes_data(args, device='cuda', is_val=False)
    # get_humanise_scenes_data(args, device='cuda', is_val=True)
    # get_humanise_scenes_data(args, device='cuda', is_val=True, scene_norm=False)
    # get_humanise_scenes_data(args, device='cuda', is_val=True, **scene_opts)
    # get_prox_scenes_data(args, device='cuda')
    get_prox_scenes_data(args, device='cuda', **scene_opts)
