# Copyright (C) 2022-2023 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import pytorch3d
import pytorch3d.utils
import pytorch3d.renderer
import pickle
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.body_model.body_model import BodyModel
from PIL import Image, ImageDraw, ImageFont
from utils.threed.geometry import *
import sys
from utils.threed.geometry import find_best_camera_for_video
from utils.render_utils import set_renderer

import os
import os.path as osp
import numpy as np
import trimesh
# os.environ["PYOPENGL_PLATFORM"] = "egl"
import torch
from pathlib import Path

from pytorch3d.structures import join_meshes_as_scene
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from tqdm import tqdm
from utils.misc import save_trimesh
from utils.misc import read_pickle
from utils.misc import read_json
# from pyquaternion import Quaternion as Q
from utils.pyquaternion import Quaternion as Q
from math import pi
from utils.threed.scene_data import rot_matrix

class PyTorch3DRenderer(torch.nn.Module):
    """
    Thin wrapper around pytorch3d threed.
    Only square renderings are supported.
    Remark: PyTorch3D uses a camera convention with z going out of the camera and x pointing left.
    """

    def __init__(self,
                 image_size,
                 background_color=(0, 0, 0),
                 convention='opencv',
                 blur_radius=0,
                 faces_per_pixel=1,
                 bg_blending_radius=0,
                 ):
        super().__init__()
        self.image_size = image_size

        raster_settings_soft = pytorch3d.renderer.RasterizationSettings(
            image_size=image_size,
            blur_radius=blur_radius,
            faces_per_pixel=faces_per_pixel)
        rasterizer = pytorch3d.renderer.MeshRasterizer(raster_settings=raster_settings_soft)

        materials = pytorch3d.renderer.materials.Materials(shininess=1.0)
        blend_params = pytorch3d.renderer.BlendParams(background_color=background_color)

        # One need to attribute a camera to the shader, otherwise the method "to" does not work.
        dummy_cameras = pytorch3d.renderer.OrthographicCameras()
        shader = pytorch3d.renderer.SoftPhongShader(cameras=dummy_cameras,
                                                    materials=materials,
                                                    blend_params=blend_params)

        # Differentiable soft threed using per vertex RGB colors for texture
        self.renderer = pytorch3d.renderer.MeshRenderer(rasterizer=rasterizer, shader=shader)

        self.convention = convention
        if convention == 'opencv':
            # Base camera rotation
            base_rotation = torch.as_tensor([[[-1, 0, 0],
                                              [0, -1, 0],
                                              [0, 0, 1]]], dtype=torch.float)
            self.register_buffer("base_rotation", base_rotation)
            self.register_buffer("base_rotation2d", base_rotation[:, 0:2, 0:2])

        # Light Color
        self.ambient_color = 0.5
        self.diffuse_color = 0.3
        self.specular_color = 0.2

        self.bg_blending_radius = bg_blending_radius
        if bg_blending_radius > 0:
            self.register_buffer("bg_blending_kernel",
                                 2.0 * torch.ones((1, 1, 2 * bg_blending_radius + 1, 2 * bg_blending_radius + 1)) / (
                                         2 * bg_blending_radius + 1) ** 2)
            self.register_buffer("bg_blending_bias", -torch.ones(1))
        else:
            self.blending_kernel = None
            self.blending_bias = None

    def to(self, device):
        # Transfer to device is a bit bugged in pytorch3d, one needs to do this manually
        self.renderer.shader.to(device)
        return super().to(device)

    def render(self, vertices, faces, cameras, color=None):
        """
        Args:
            - vertices: [B,N,V,3]
            - faces: [B,F,3]
            - maps: [B,N,W,H,3] in 0-1 range - if None the texture will be metallic
            - cameras: PerspectiveCamera or OrthographicCamera object
            - color: [B,N,V,3]
        Return:
            - img: [B,W,H,C]
        """

        if isinstance(vertices, torch.Tensor):
            _, N, V, _ = vertices.size()
            list_faces = []
            list_vertices = []
            for i in range(N):
                list_faces.append(faces + V * i)
                list_vertices.append(vertices[:, i])
            faces = torch.cat(list_faces, 1)  # [B,N*F,3]
            vertices = torch.cat(list_vertices, 1)  # [B,N*V,3]

            # Metallic texture
            verts_rgb = torch.ones_like(vertices).reshape(-1, N, V, 3)  # [1,N,V,3]
            if color is not None:
                verts_rgb = color * verts_rgb
            verts_rgb = verts_rgb.flatten(1, 2)
            
            textures = pytorch3d.renderer.Textures(verts_rgb=verts_rgb)
            # Create meshes
            meshes = pytorch3d.structures.Meshes(verts=vertices, faces=faces , textures=textures)
        else:
            tex = [torch.ones_like(vertices[i]) * color[i] for i in range(len(vertices))]
            tex = torch.cat(tex)[None]
            textures = pytorch3d.renderer.Textures(verts_rgb=tex)
            
            verts = torch.cat(vertices)

            faces_up = []
            n = 0
            for i in range(len(faces)):
                faces_i = faces[i] + n
                faces_up.append(faces_i)
                n += vertices[i].shape[0]
            faces = torch.cat(faces_up)
            # ipdb.set_trace()
            meshes = pytorch3d.structures.Meshes(verts=[verts], faces=[faces], textures=textures)

        # Create light
        lights = pytorch3d.renderer.DirectionalLights(
            ambient_color=((self.ambient_color, self.ambient_color, self.ambient_color),),
            diffuse_color=((self.diffuse_color, self.diffuse_color, self.diffuse_color),),
            specular_color=(
                (self.specular_color, self.specular_color, self.specular_color),),
            direction=((0, 0, -1.0),),
            device=vertices[0].device)
        images = self.renderer(meshes, cameras=cameras, lights=lights)

        rgb_images = images[..., :3]
        rgb_images = torch.clamp(rgb_images, 0., 1.)
        rgb_images = rgb_images * 255
        rgb_images = rgb_images.to(torch.uint8)

        return rgb_images

    def renderPerspective(self, vertices, faces, camera_translation, principal_point=None, color=None, rotation=None,
                          focal_length=2 * FOCAL_LENGTH / IMG_SIZE):
        """
        Args:
            - vertices: [B,V,3] or [B,N,V,3] where N is the number of persons
            - faces: [B,13776,3]
            - focal_length: float
            - principal_point: [B,2]
            - T: [B,3]
            - color: [B,N,3]
        Return:
            - img: [B,W,H,C] in range 0-1
        """

        device = vertices[0].device
        # device = vertices.device

        if principal_point is None:
            principal_point = torch.zeros_like(camera_translation[:, :2])

        if isinstance(vertices, torch.Tensor) and vertices.dim() == 3:
            vertices = vertices.unsqueeze(1)

        # Create cameras
        if rotation is None:
            R = self.base_rotation
        else:
            R = torch.bmm(self.base_rotation, rotation)
        camera_translation = torch.einsum('bik, bk -> bi', self.base_rotation.repeat(camera_translation.size(0), 1, 1),
                                          camera_translation)
        if self.convention == 'opencv':
            principal_point = -torch.as_tensor(principal_point)
        cameras = pytorch3d.renderer.PerspectiveCameras(focal_length=focal_length, principal_point=principal_point,
                                                        R=R, T=camera_translation, device=device)

        rgb_images = self.render(vertices, faces, cameras, color)

        return rgb_images


def torch_mesh_from_body(vertices, faces, device='cpu', color=None):
    if not isinstance(faces, torch.Tensor):
        faces = torch.tensor(faces, dtype=torch.int64)
    faces = faces.float()
    V = vertices.size(0)
    if color is None:
        # this is light blue
        color = (torch.tensor([182, 249, 239])/255.).expand(1, V, 3).to(device)
    tex = TexturesVertex(verts_features=color)
    mesh_scene = Meshes(verts=vertices[None], faces=faces[None], textures=tex).to(device)
    return mesh_scene


def torch_mesh_from_trimesh(static_scene, is_body=False, device='cpu'):
    import matplotlib.pyplot as plt

    # static_scene = trimesh.load(obj_filename)
    vertices = torch.from_numpy(static_scene.vertices).to(device).type(torch.float32)
    faces = torch.from_numpy(static_scene.faces).to(device).type(torch.float32)

    if is_body:
        V = vertices.size(0)
        body_c = plt.get_cmap('viridis', lut=V)(range(V))[:,:3]
        color = torch.from_numpy(body_c).float().reshape(1, V, 3).to(device)
    else:
        color = torch.from_numpy(static_scene.visual.vertex_colors).to(device).type(torch.float32)[None, :, :3] / 255.

    tex = TexturesVertex(verts_features=color)
    mesh_scene = Meshes(verts=vertices[None], faces=faces[None], textures=tex).to(device)
    return mesh_scene


def read_scene_mesh(scene_name, rot=None, return_trans=False, dataset='prox', norm_scene=False):
    T4 = Q(axis=[0, 0, 1], angle=pi/2.).transformation_matrix.astype(np.float32)
    T5 = Q(axis=[1, 0, 0], angle=pi/2.).transformation_matrix.astype(np.float32)
    T = T5 @ T4
    """reads scene mesh from data folder"""
    if dataset=='prox':
        base_dir = Path('./data/prox_with_lemo')
        cam2world_dir = osp.join(base_dir, 'cam2world')
        scene_dir = osp.join(base_dir, 'scenes_downsampled')

        static_scene = trimesh.load(osp.join(scene_dir, scene_name + '.ply'), process=False)
        cam = read_json(os.path.join(cam2world_dir, scene_name + '.json'))
        cam = np.array(cam)
        trans = np.linalg.inv(cam)

        ys_path = cam2world_dir / Path(scene_name + '_transl.json')
        ys = read_json(ys_path)
        max_y_mean = ys['max_y_mean']

        if rot is not None:
            rot = rot.clone().detach().cpu().numpy()
            static_scene.vertices = static_scene.vertices @ rot[0]

        if norm_scene:
            static_scene.apply_transform(rot_matrix)
            scene_verts = static_scene.vertices
            scene_verts[:, 1] -= max_y_mean
            static_scene.vertices = scene_verts
            # from utils.misc import save_trimesh
            # path = Path('inspect_out/images/visu_generations/debug/scene_norm.obj')
            # path.parent.mkdir(parents=True, exist_ok=True)
            # save_trimesh(static_scene.vertices, static_scene.faces, path)
        else:
            static_scene = static_scene.apply_transform(trans)
            # from utils.misc import save_trimesh
            # path = Path('inspect_out/images/visu_generations/debug/scene_transf.obj')
            # path.parent.mkdir(parents=True, exist_ok=True)
            # save_trimesh(static_scene.vertices, static_scene.faces, path)

        if return_trans:
            return static_scene, trans
        else:
            return static_scene, cam, max_y_mean

    elif dataset=='humanise':
        base_dir = Path('./data/scannet/scans')
        # scene_dir = osp.join(base_dir, f'{scene_name}/{scene_name}_vh_clean_2.ply')
        scene_dir = osp.join(base_dir, f'{scene_name}/{scene_name}_vh_clean_2_no_walls.ply')
        static_scene = trimesh.load(scene_dir, process=False)
        transl_file = './data/humanise/scene_translations.pkl'
        t_data = read_pickle(transl_file)[scene_name.split('_')[0]]
        static_scene.apply_translation(t_data[0])
        static_scene.apply_transform(T)
        return static_scene, None, None
    else:
        raise NotImplementedError



def render_video_w_scene(scene_name, meta, vertices, faces, cameras=None, image_size=400, add_border=True,
                         text=None, pad_width=None, rotation=None, color=None, adapt_camera=False,
                         return_scene=False,
                         rot=None, verbose=False, static_scene=None, act_labels=None, norm_scene=True, unorm_body=True,
                         trasl=False,
                         ):
    """
    Rendering human 3d mesh into RGB images
    :param verts: tensor of shape [seq_len,V,3]
    :param faces: [1,13776,3]
    :param camera_translation: [seq_len,3]
    :param image_size: int
    :param device: cpu or cuda
    :param color: [seq_len,N,V,3] or list of [N,V,3] of length 'seq_len'
    :return: video: [seq_len,image_size,image_size,3]
    """
    r_cons, g_cons, b_cons = 0, 255, 0
    pad_width = 0 if pad_width is not None else int(0.025 * image_size)
    font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 12)

    if adapt_camera:
        cam = find_best_camera_for_video(vertices.cpu(), factor=0.9, n_jts_to_take=256)
        vertices = vertices + cam[None].cuda()
        # cameras[:] = cam
        # from utils.misc import save_mesh
        # path = Path('inspect_out/meshes/visu')
        # path.mkdir(parents=True, exist_ok=True)
        # save_mesh(vertices.cpu()[:2], str(path/Path('seq_orig.obj')), faces[0].cpu())
        # verts_cam = vertices.cpu() + cam[None]
        # save_mesh(verts_cam.cpu()[:2], str(path/Path('seq_cam.obj')), faces[0].cpu())

    device = 'cuda'
    dataset = meta['dataset'][0]
    with torch.no_grad():
        renderer = set_renderer(device, focal=1.8, xtrans=-0.7, ytrans=0.5, ztrans=1.0, W=image_size,
                                lights_location=[[1.0, 0.0, -3.0]], dataset=dataset)
        # renderer = set_renderer(device, focal=1.8, xtrans=-1.2, ytrans=0.5, ztrans=0.8, W=400)
        B, V, _ = vertices.shape
        sub_batch = 16
        total_batch_size = vertices.shape[0] #// 2
        per_batch = total_batch_size // sub_batch
        last = total_batch_size % sub_batch

        # if given do not read
        if static_scene is None:
            static_scene, cam_sc, y_trans = read_scene_mesh(scene_name, rot, dataset=dataset,
                                                            norm_scene=norm_scene)
            if adapt_camera:
                static_scene.vertices = static_scene.vertices + cam.cpu().numpy()
            if dataset=='prox':
                T1 = Q(axis=[1, 0, 0], angle=pi / 2.).transformation_matrix.astype(np.float32)
                T2 = np.eye(4).astype(np.float32)
                # if trasl:
                T2[1, 3] = -y_trans

                T = T1 @ cam_sc
                T_inv = torch.from_numpy(np.linalg.inv(T)).float().to(vertices.device)
                # if norm_scene and unorm_body:
                if unorm_body:
                    # with the correction on the data files, this should work. The new generations already give norm body
                    vertices_orig = vertices - torch.from_numpy(T2[:3, 3][None, None]).float().to(vertices.device)
                    vertices_cam = (T_inv[None, :3, :3] @ vertices_orig.permute(0, 2, 1)).permute(0, 2, 1) + T_inv[:3, 3][None, None]
                    vertices = vertices_cam

            if 0:
                path = Path(f'inspect_out/images/visu_generations/debug/{scene_name}')
                path.mkdir(parents=True, exist_ok=True)
                # save_trimesh(vertices[0], faces[0], str(path / Path('verts_org.obj')))
                save_trimesh(vertices[0], faces[0], str(path / Path('verts_org.ply')))
                static_scene.export(str(path / Path('scene_org.obj')));
            if 0:
                path = Path(f'inspect_out/meshes/tb_visu/{scene_name}')
                path = Path(f'inspect_out/images/visu_generations/debug/{scene_name}')
                path.mkdir(parents=True, exist_ok=True)
                # static_scene.export(str(path/Path('scene_cam.obj')));
                static_scene.export(str(path/Path('scene_cam.ply')));
                save_trimesh(vertices_[0], faces[0], str(path/Path('seq_cam.obj')))
                save_trimesh(vertices_orig[0], faces[0], str(path/Path('body_orig_space.obj')))
                # save_trimesh(vertices[0], faces[0], str(path/Path('body_nomr_gt.obj')))
                save_trimesh(vertices[0], faces[0], str(path/Path('body_nomr_gt.ply')))
                save_trimesh(vertices_cam[0], faces[0], str(path/Path('body_orig_cam_space.obj')))

        fB = faces.shape[0]
        B = vertices.shape[0]

        if fB != B and fB == 1:
            if isinstance(faces, np.ndarray):
                faces = torch.tensor(faces)
            faces = faces.expand(B, -1, -1)

        all_images = []
        for i in tqdm(range(0, per_batch + 1), disable=not verbose):

            verts_list = []
            faces_list = []
            texture_list = []
            if i == per_batch:
                iter_this = last
            else:
                iter_this = sub_batch

            for j in range(iter_this):
                index = i*sub_batch + j
                color_ = color[index, None].to(device) if color is not None else None
                mesh_body = torch_mesh_from_body(vertices[index], faces[index], device=device, color=color_)
                mesh_scene = torch_mesh_from_trimesh(static_scene, is_body=False, device=device)
                mesh = join_meshes_as_scene([mesh_scene, mesh_body])
                verts_list.append(mesh.verts_padded()[0])
                faces_list.append(mesh.faces_padded()[0]) # for HUMANISE ~160K faces
                texture_list.append(mesh.textures._verts_features_list[0])
                if 0:
                    path = Path(f'inspect_out/images/visu_generations/debug/{scene_name}')
                    path.mkdir(parents=True, exist_ok=True)
                    # save_trimesh(verts_list[0], faces_list[0], str(path / Path('before_rend.obj')))
                    save_trimesh(verts_list[0], faces_list[0], str(path / Path('before_rend.ply')))



            if len(verts_list) > 0:
                tex = TexturesVertex(verts_features=texture_list)
                meshes_joint = Meshes(verts=verts_list, faces=faces_list, textures=tex)
                images_ = renderer(meshes_joint).cpu()
                images = (255 * images_[..., :3]).to(torch.uint8)
                all_images.append(images)
                # plot(images[0])
                # plot(images[0, ..., :3])

    del renderer
    torch.cuda.empty_cache()
    video = np.concatenate(all_images)

    # add text
    img_w_text = []
    if text is not None:
        for idx, image in enumerate(video):
            ids = f'{idx:06d}'
            # image = image.cpu().numpy()
            if image.mean() < 5:
                img = (255 * image[..., :3]).astype(np.uint8)
            else:
                img = (image[..., :3]).astype(np.uint8)
            img = Image.fromarray(img)
            draw = ImageDraw.Draw(img)
            draw.text((1.5 * pad_width, 1.3 * pad_width), text, fill=(128, 128, 128), font=font)
            draw.text((1.5 * pad_width, 0 * pad_width), ids, fill=(128, 128, 128), font=font)
            if act_labels is not None:
                draw.text((1.5 * pad_width, 2.5 * pad_width), act_labels[idx], fill=(128, 128, 128), font=font)
            image = np.asarray(img)
            img_w_text.append(image)
        video = np.stack(img_w_text)
        # plot(video[0])


    # add border
    img_w_border = []
    if add_border:
        for image in video:
            rb = np.pad(array=image[..., 0], pad_width=pad_width, mode='constant', constant_values=r_cons)
            gb = np.pad(array=image[..., 1], pad_width=pad_width, mode='constant', constant_values=g_cons)
            bb = np.pad(array=image[..., 2], pad_width=pad_width, mode='constant', constant_values=b_cons)
            img_w_border.append(np.dstack(tup=(rb, gb, bb)))
        video = np.stack(img_w_border)
        # plot(img_w_border[0])


    if return_scene:
        return video, static_scene
    else:
        return video



def render_image_w_several_poses(scene_name, meta, vertices, faces, cameras=None, image_size=400, add_border=True,
                         text=None, pad_width=None, rotation=None, color=None, adapt_camera=False, return_scene=False,
                         rot=None, verbose=False, static_scene=None, act_labels=None, gamma=0.014
                         ):
    """
    Rendering human 3d mesh into RGB images
    :param verts: tensor of shape [seq_len,V,3]
    :param faces: [1,13776,3]
    :param camera_translation: [seq_len,3]
    :param image_size: int
    :param device: cpu or cuda
    :param color: [seq_len,N,V,3] or list of [N,V,3] of length 'seq_len'
    :return: video: [seq_len,image_size,image_size,3]
    """
    r_cons, g_cons, b_cons = 0, 255, 0
    pad_width = 0 if pad_width is not None else int(0.025 * image_size)
    font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 12)

    device = 'cuda'
    with torch.no_grad():
        renderer = set_renderer(device, focal=1.8, xtrans=-0.5, ytrans=0.5, ztrans=0.8, W=400, gamma=gamma)

        B, V, _ = vertices.shape

        # if given do not read
        if static_scene is None:
            static_scene, _, _ = read_scene_mesh(scene_name, rot)

        all_images = []

        verts_list = []
        faces_list = []
        texture_list = []


        index = 0
        all_body_meshes = []
        for index in range(len(vertices)):
            color_ = color[index, None] if color is not None else None
            mesh_body = torch_mesh_from_body(vertices[index], faces[0], device=device, color=color_)
            all_body_meshes.append(mesh_body)

        mesh_scene = torch_mesh_from_trimesh(static_scene, is_body=False, device=device)
        # merge all body meshes + scene
        mesh = join_meshes_as_scene([mesh_scene, *all_body_meshes])
        verts_list.append(mesh.verts_padded()[0])
        faces_list.append(mesh.faces_padded()[0])
        texture_list.append(mesh.textures._verts_features_list[0])

        if len(verts_list) > 0:
            tex = TexturesVertex(verts_features=texture_list)
            meshes_joint = Meshes(verts=verts_list, faces=faces_list, textures=tex)
            image = renderer(meshes_joint).cpu()
            # all_images.append(images)
            # plot(images[0])
            # plot(images[0, ..., :3])

    return image

def test():
    import matplotlib.pyplot as plt

    img_size = 500
    f_x = f_y = FOCAL_LENGTH

    # Load a AMASS SMPL pose
    fn = '/tmp-network/user/fbaradel/projects/HumanPoseGeneration/data/mocap/smplh/amass/SFU/0005/0005_Jogging001_poses.pkl'
    with open(fn, 'rb') as f:
        data = pickle.load(f)
    t = int(data['root_orient'].size(0) / 2.)
    root_orient = data['root_orient'][[t]]
    pose_body = data['pose_body'][[t]]
    pose_hand = data['pose_hand'][[t]]
    trans = data['trans'][[t]]
    trans = torch.zeros_like(trans)

    # Creating a human mesh
    bm = BodyModel(bm_fname='/tmp-network/SlowHA/user/fbaradel/data/SMPLX/smplh/neutral/model.npz')
    bm_out = bm(root_orient=root_orient, pose_body=pose_body, pose_hand=pose_hand, trans=trans)
    vertices = bm_out.v
    joints = bm_out.Jtr

    # Find best camera params
    camera_translation = find_best_camera(joints, factor=1.3, f_x=f_x, f_y=f_y)

    # Rendering
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    rend = PyTorch3DRenderer(img_size).to(device)
    pyfaces = torch.as_tensor(np.array(c2c(bm.f), dtype=np.int32), device=device)[None, :, :]
    vertices = torch.stack([vertices], 1)  # [t,n,v,3]

    V = vertices.size(2)
    color = plt.get_cmap('viridis', lut=V)(range(V))[:,:3]
    color = torch.from_numpy(color).float().reshape(1, 1, V, 3)

    with torch.no_grad():
        img = rend.renderPerspective(vertices=vertices.to(device),
                                     faces=pyfaces.to(device),
                                     camera_translation=camera_translation.to(device),
                                     color=color.to(device)
                                     )[0].cpu().numpy()
    image = Image.fromarray(img)
    image.save('img.jpg')


def test_renderer():
    img_size = 500
    f_x = f_y = FOCAL_LENGTH

    # Load a AMASS SMPL pose
    fn = '/tmp-network/user/fbaradel/projects/HumanPoseGeneration/data/mocap/smplh/amass/SFU/0005/0005_Jogging001_poses.pkl'
    with open(fn, 'rb') as f:
        data = pickle.load(f)
    t = int(data['root_orient'].size(0) / 2.)
    root_orient = data['root_orient'][[t]]
    pose_body = data['pose_body'][[t]]
    pose_hand = data['pose_hand'][[t]]
    trans = data['trans'][[t]]
    trans = torch.zeros_like(trans)
    print(trans)

    # Creating a human mesh
    bm = BodyModel(bm_fname='/tmp-network/SlowHA/user/fbaradel/data/SMPLX/smplh/neutral/model.npz')
    bm_out = bm(root_orient=root_orient, pose_body=pose_body, pose_hand=pose_hand, trans=trans)
    vertices = bm_out.v
    joints = bm_out.Jtr

    # Find best camera params
    camera_translation = find_best_camera(joints, factor=1.3, f_x=f_x, f_y=f_y)

    # Rendering
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    rend = PyTorch3DRenderer(img_size).to(device)
    pyfaces = torch.as_tensor(np.array(c2c(bm.f), dtype=np.int32), device=device)[None, :, :]
    vertices = torch.stack([vertices, 0.25 + vertices], 1)  # [t,n,V,3] - add the same person
    with torch.no_grad():
        img = rend.renderPerspective(vertices=vertices.to(device),
                                     faces=pyfaces.to(device),
                                     camera_translation=camera_translation.to(device),
                                     color=torch.randn(1, 2, 3).to(device)
                                     )[0].cpu().numpy()
    image = Image.fromarray(img)

    # Project jts into 2d
    camera_center = torch.as_tensor([[IMG_SIZE / 2., IMG_SIZE / 2.]])
    rotation = torch.eye(3).type_as(joints).unsqueeze(0)
    keypoints = perspective_projection(world2cam(joints, camera_translation, rotation), camera_center, f_x, f_y)
    keypoints /= IMG_SIZE
    draw = ImageDraw.Draw(image)
    r = 2
    for po in keypoints[0]:
        x_, y_ = po * img_size
        leftUpPoint = (x_ - r, y_ - r)
        rightDownPoint = (x_ + r, y_ + r)
        twoPointList = [leftUpPoint, rightDownPoint]
        draw.ellipse(twoPointList, fill='red')
    image.save('img.jpg')

    # Estim translation
    camera_translation_hat = estimate_translation_np(joints[0].numpy(), keypoints[0].numpy() * IMG_SIZE,
                                                     f_x=f_x, f_y=f_y)
    print(camera_translation_hat, camera_translation)
    for i in range(3):
        assert abs(camera_translation_hat[i] - camera_translation[0, i]).item() < 0.0001

    # import ipdb
    # ipdb.set_trace()
    out = render_video(vertices, camera_translation, pyfaces, device=None)
    Image.fromarray(out[0]).save(f"img_bis.jpg")


def test_video_rendering():
    from tqdm import tqdm
    import os

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Load a AMASS SMPL pose
    fn = '/tmp-network/user/fbaradel/projects/HumanPoseGeneration/data/mocap/amass/SFU/0005/0005_Jogging001_poses.pkl'
    with open(fn, 'rb') as f:
        data = pickle.load(f)

    # Creating a human mesh
    print("Generating 3d human mesh using SMPL")
    seq_len = 60
    bm = BodyModel(bm_fname='/tmp-network/SlowHA/user/fbaradel/data/SMPLX/smplh/neutral/model.npz')
    trans = data['trans'][:seq_len]
    # trans = trans - trans[[int(seq_len/2.)]]
    trans = trans - trans[[0]]
    bm_out = bm(root_orient=data['root_orient'][:seq_len], pose_body=data['pose_body'][:seq_len],
                pose_hand=data['pose_hand'][:seq_len],
                trans=trans)
    vertices = bm_out.v
    joints = bm_out.Jtr

    # Find best camera params
    print("Finding the best camera params")
    camera_translation = find_best_camera_for_video(joints, factor=1.3, n_jts_to_take=100)
    print(camera_translation)
    camera_translation = camera_translation.repeat(seq_len, 1)

    # Rendering
    print("2D rendering")
    pyfaces = torch.as_tensor(np.array(c2c(bm.f), dtype=np.int32), device=device)[None, :, :]
    print(vertices.shape, camera_translation.shape, pyfaces.shape)
    video = render_video(vertices, camera_translation, pyfaces, last_t_green_border=int(seq_len / 2.), text='Test')

    # Building video
    print("Video creation")
    tmp_dir = 'output'
    for t in tqdm(range(video.shape[0])):
        Image.fromarray(video[t]).save(f"{tmp_dir}/{t:06d}.jpg")
    cmd = f"ffmpeg -framerate 10 -pattern_type glob -i '{tmp_dir}/*.jpg' -c:v libx264 -vf fps=30 -pix_fmt yuv420p {tmp_dir}/video.mp4 -y"
    os.system(cmd)
    os.system(f"rm {tmp_dir}/*.jpg")


def test_camera_params_estimation():
    import os

    # Load a AMASS SMPL pose
    fn = '/tmp-network/user/fbaradel/projects/HumanPoseGeneration/data/mocap/amass/SFU/0005/0005_Jogging001_poses.pkl'
    with open(fn, 'rb') as f:
        data = pickle.load(f)
    seq_len = 30
    factor = 1.1
    ttt = [int(data['root_orient'].size(0) / 2.) + i for i in range(seq_len)]
    root_orient = data['root_orient'][ttt]
    pose_body = data['pose_body'][ttt]
    pose_hand = data['pose_hand'][ttt]
    trans = data['trans'][ttt]
    trans = trans - trans[[0]]

    # Creating a human mesh
    bm = BodyModel(bm_fname='/tmp-network/SlowHA/user/fbaradel/data/SMPLX/smplh/neutral/model.npz')
    faces = torch.as_tensor(np.array(c2c(bm.f), dtype=np.int32))[None, :, :]
    bm_out = bm(root_orient=root_orient, pose_body=pose_body, pose_hand=pose_hand)

    # Rendering using trans
    camera = find_best_camera_for_video(bm_out.Jtr + trans.unsqueeze(1), factor=factor)
    print("Camera: ", camera)
    video = render_video(bm_out.v + trans.unsqueeze(1), camera.repeat(seq_len, 1), faces, text='w/ trans')

    # Estim. camera
    camera_bis = estimate_video_camera_params_wo_trans(bm_out.Jtr.unsqueeze(0), trans.unsqueeze(0), factor=factor)[0]
    print("Camera bis t=0: ", camera_bis[0])
    video_bis = render_video(bm_out.v, camera_bis, faces, text='wo trans')

    print(np.abs(video_bis - video).sum())

    # Create video
    visu_dir = './output'
    os.makedirs(visu_dir, exist_ok=True)
    for t in range(seq_len):
        img = np.concatenate([video[t], video_bis[t]], 1)
        Image.fromarray(img).save(f"{visu_dir}/{t:06d}.jpg")
    cmd = f"ffmpeg -framerate 5 -pattern_type glob -i '{visu_dir}/*.jpg' -c:v libx264 -vf fps=30 -pix_fmt yuv420p video.mp4 -y"
    os.system(cmd)
    os.system(f"rm {visu_dir}/*.jpg")

    # print(camera)
    #
    # # Image with trans
    # cameras = torch.Tensor([[0., 0., 44.]])
    # img = render_video(bm_out.v, cameras, faces)[0]
    #
    # # Image w/o trans but with estimated cam params
    # kps = perspective_projection(bm_out.Jtr, cameras)  # range 0-1
    # jts = bm_out.Jtr - trans
    # cameras_hat = estimate_translation_np(jts[0].numpy(), kps[0].numpy())
    # cameras_hat = torch.from_numpy(cameras_hat).unsqueeze(0).float()
    # img_bis = render_video(bm_out.v - trans, cameras_hat, faces)[0]
    # Image.fromarray(np.concatenate([img, img_bis])).save(f"img.jpg")
    # print(np.abs(img_bis - img).sum())


if __name__ == "__main__":
    exec(sys.argv[1])
