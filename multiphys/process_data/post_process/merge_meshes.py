import numpy as np
from pathlib import Path
import trimesh
from utils.misc import save_mesh
from tqdm import tqdm, trange

def get_verts_faces(path):
    mesh = trimesh.load(path, process=False)
    verts = mesh.vertices
    faces = mesh.faces
    return verts, faces

def main():
    path_p1 = "results/scene+/tcn_voxel_4_5_chi3d_phalp/results/chi3d/s02_Grab_1/2023-07-18_11-55-33/meshes"
    path_p2 = "results/scene+/tcn_voxel_4_5_chi3d_phalp/results/chi3d/s02_Grab_1/2023-07-18_11-59-15/meshes"
    root = Path(path_p1).parent
    files_p1 = sorted(list(Path(path_p1).glob("*.ply")))
    files_p2 = sorted(list(Path(path_p2).glob("*.ply")))

    for n in trange(len(files_p1)):
        file1 = files_p1[n]
        file2 = files_p2[n]
        verts, faces = get_verts_faces(file1)
        verts2, faces = get_verts_faces(file2)
        vertices = [verts, verts2]

        save_mesh(vertices, faces=faces, out_path=f"{root}/meshes_merged/verts_{n:04d}.ply")


if __name__ == "__main__":
    main()
