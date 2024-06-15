import numpy as np
import trimesh
from pathlib import Path
from utils.misc import create_dir
from utils.misc import save_mesh

color_dict = {
    'white':[1., 1., 1.],
    'blue':[0, 0., 1.],
    'green':[0, 1., 0.],
    'black': [0, 0., 0.],
    'blackish':[0.349, 0.270, 0.270],
    'orange_light':[0.886, 0.654, 0.431],
    'cherry_red':[0.756, 0.121, 0.168],
    'light_blue':[0.215, 0.701, 0.701],
    'marino':[0.215, 0.435, 0.701],
    'light_yellow':[0.878, 0.882, 0.576],
    # 'sky':[0.305, 0.607, 0.815],
    'sky':[0.447, 0.678, 0.835],
}

R_UPPER_IDXS = [2, 3, 4]
L_UPPER_IDXS = [5, 6, 7]
R_LOWER_IDXS = [8, 9, 10]
L_LOWER_IDXS = [11, 12, 13]

def create_skel_spheres(j3d, radius=0.04, color=[0, 0, 1.]):
    # creates spheres from joints for one person sin input dims=(17,3)
    jts_spheres = []
    for n,xyz in enumerate(j3d):
        if n==4 or n==7:
            radius_ = radius +0.01
        else:
            radius_ = radius
        sphere = trimesh.primitives.Sphere(radius=radius_, center=xyz)
        if n in L_UPPER_IDXS:
            color_ = color_dict['orange_light']
        elif n in R_UPPER_IDXS:
            color_ = color_dict['cherry_red']
        elif n in R_LOWER_IDXS:
            color_ = color_dict['marino']
        elif n in L_LOWER_IDXS:
            color_ = color_dict['light_blue']
        else:
            color_ = color_dict['light_yellow']
            # color_ = color_dict['sky']
        sphere.visual.face_colors = color_ + [0.5]
        jts_spheres.append(sphere)
    # save_mesh(jts_spheres, './tests_paper/sphere.obj')
    return jts_spheres


def get_distance(points):
    p1, p2 = points
    p_diff = p1 - p2
    p_sqr = p_diff * p_diff
    p_sqr_sum = p_sqr.sum()
    norm = np.sqrt(p_sqr_sum)
    return norm


def create_cylinder(height, radius=0.002, color = [0, 0, 1.]):
    # radius = 0.002
    new_height = height - 2*radius
    cylinder = trimesh.creation.cylinder(radius=radius, height=new_height)
    # cylinder.visual.face_colors = [0, 0, 1., 0.5]
    color = (255*np.array(color)).astype(np.uint8)
    cylinder.visual.face_colors = color.tolist() + [255]
    translation = [0., 0., new_height/2.+ radius]  # cylinder offset + plane offset
    cylinder.apply_translation(translation)
    return cylinder


def gen_cylinder():
    # scene = trimesh.Scene()
    # object-2 (cylinder)
    cylinder = trimesh.creation.cylinder(radius=0.01, height=0.3)
    cylinder.visual.face_colors = [0, 0, 1., 0.5]
    # axis = trimesh.creation.axis(origin_color=[1., 0, 0])
    translation = [0.1, -0.2, 0.15 + 0.01]  # cylinder offset + plane offset
    cylinder.apply_translation(translation)
    # axis.apply_translation(translation)
    # scene.add_geometry(cylinder)
    # scene.add_geometry(axis)
    # scene.export('./cylinder.ply');
    cylinder.export('./cylinder.ply');


def save_mesh_from_trimesh(tri_meshes, out_path='scene.obj'):
    create_dir(Path(out_path).parent)
    scene = tri_meshes[0].scene()
    for v in tri_meshes[1:]:
        scene.add_geometry(v)
    # scene.show()
    scene.export(out_path)


def save_one_trimesh(one_trimesh, out_path='scene.obj'):
    create_dir(Path(out_path).parent)
    one_trimesh.export(out_path)

def joints_to_skel(j3d_all,
                   out_file,
                   radius=0.02,
                   sphere_rad=0.03,
                   format='mupots',
                   color_idx=0,
                   save_jts=False,
                   mask=None,
                   ):


    """
    This is in MuPots joints format
    :param j3d_all: it is a np array containing several persons dims have to be (Np, Njts, 3)
    :param out_path: has to be in Path format Posix-path
    :param radius:
    :param name:
    :return:
    """
    if format=='mupots':
        order0 = [0, 16, 1, 2, 3, 1, 5, 6, 1, 15, 14, 11, 12, 14, 8, 9]
        order1 = [16, 1, 2, 3, 4, 5, 6, 7, 15, 14, 11, 12, 13, 8, 9, 10]
    elif format=='NBA':
        # for basketball dataset this orders
        order0 = [0,0,1,2,4,5,0,7,8,8,13,8,10,11,14,]
        order1 = [1,4,2,3,5,6,7,8,9,13,14,10,11,12,15,]
    elif format=='expi':
        # for basketball dataset this orders
        order0 = [0, 0, 0, 3, 3,  3,  3, 4, 6, 5, 7, 10, 12, 14, 11, 13, 15]
        order1 = [1, 2, 3, 4, 5, 10, 11, 6, 8, 7, 9, 12, 14, 16, 13, 15, 17]
    elif format=='smpl':
        order0 = [0, 0, 0, 2, 5, 8,  1, 4,  7, 3, 6,  9, 12, 12, 17, 19, 12, 16, 18]
        order1 = [1, 2, 3, 5, 8, 11, 4, 7, 10, 6, 9, 12, 15, 17, 19, 21, 16, 18, 20]
    elif format=='smpl_red':
        order0 = [0, 0, 1, 3, 5, 2, 4, 6, 10, 12, 9,  11]
        order1 = [1, 2, 3, 5, 7, 4, 6, 8, 12, 14, 11, 13]
    else:
        print('Not implemented, must choose a correct JOINT FORMAT!')
        return

    connections = [[c, z] for c, z in zip(order0, order1)]

    white = [1., 1., 1.]
    blue = [0, 0., 1.]
    green = [0, 1., 0.]
    black = [0, 0., 0.]
    blackish = [0.349, 0.270, 0.270]
    orange_light = [0.886, 0.654, 0.431]

    colors = [white, blue, green, black, blackish, orange_light]
    color = colors[color_idx]

    if mask is not None:
        masked_jts = np.where(mask == 0)[0]

    all_bones = []
    for n, this_j3d in enumerate(j3d_all):
        # guardar los joints como espferas antes de hacer root-relative
        if mask is not None:
            this_j3d_masked = this_j3d[mask.astype(bool)]
        else:
            this_j3d_masked = this_j3d
        jts_spheres = create_skel_spheres(this_j3d_masked, radius=sphere_rad * radius/0.01)

        all_bones += jts_spheres
        # Primero hacer root aligned y luego ver como hacer pose absoluta
        orig_root = this_j3d[14]
        this_j3d = this_j3d - orig_root
        j3d = this_j3d
        # bones = []


        for i, this_con in enumerate(connections):
            c1, c2 = this_con
            if mask is not None:
                if c1 in masked_jts or c2 in masked_jts:
                    continue
            # medir la longitud:
            # hip1 has two joints, array with 2 points
            hip1 = j3d[this_con]
            hip1_d = get_distance(hip1)
            cyl1 = create_cylinder(hip1_d, radius=radius, color=color)
            # alinenear el par de puntos haciendo que el punto de origen (o de partida) pase a ser
            # el origen absoluto (0, 0, 0). Luego alinear con ese angulo y trasladar al punto de partida
            # original.
            orig_strt = hip1[0]
            hip1 = hip1 - orig_strt
            # rotar como se debe, alineando con el destino
            a = trimesh.unitize(hip1[1])  # hacer el punto de destino con magnitud unitaria
            m = trimesh.geometry.align_vectors([0, 0, 1], a)
            cyl1.apply_transform(m)
            # trasladar al punto de origen
            hip1_origin_root_rel = hip1[0] + orig_strt
            hip1_origin = hip1_origin_root_rel + orig_root
            translation = hip1_origin  # cylinder offset + plane offset
            cyl1.apply_translation(translation)
            all_bones.append(cyl1)

    # out_skel_path = out_path / Path(f'{name}.ply')

    save_mesh_from_trimesh(all_bones, str(out_file))

    if save_jts:
        # filter spheres
        spheres = [c for c in all_bones if isinstance(c, trimesh.primitives.Sphere)]
        # divide in two groups
        np = j3d_all.shape[0]
        if np == 2:
            spheres_p1 = spheres[:len(spheres) // 2]
            spheres_p2 = spheres[len(spheres) // 2:]
            spheres_all = [spheres_p1, spheres_p2]
        elif np == 1:
            spheres_all = [spheres]

        for nn, spheres_n in enumerate(spheres_all):
            for n, jt in enumerate(spheres_n):
                out_file_n = Path(out_file.replace('.ply', '')) / Path(f'joint_spheres/p{nn}/jts_{n:02d}.ply')
                save_one_trimesh(jt, str(out_file_n))
        # return jts_spheres
