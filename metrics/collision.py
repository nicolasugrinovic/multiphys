import numpy as np
import trimesh
from tqdm import tqdm
from utils.smpl import smpl_to_verts
from utils.misc import save_trimesh

def split_check_contain(mesh1, v2):
    v2_ = np.split(v2, 5)
    cont_in_mesh1_ = []
    for v2_sub in v2_:
        val = mesh1.contains(v2_sub)
        cont_in_mesh1_.extend(val)
    cont_in_mesh1_ = np.array(cont_in_mesh1_)
    return cont_in_mesh1_

def check_contain(mesh1, v2):
    val = mesh1.contains(v2)
    return val


def read_verts(smpl_list):
    smpl_p1 = smpl_list[0]
    smpl_p2 = smpl_list[1]
    verts_p1, faces = smpl_to_verts(smpl_p1['pose_aa'], smpl_p1['trans'].float(), betas=smpl_p1['betas'][:, :10], return_joints=False)
    verts_p2, faces = smpl_to_verts(smpl_p2['pose_aa'], smpl_p2['trans'].float(), betas=smpl_p2['betas'][:, :10], return_joints=False)
    verts_p1 = verts_p1[0].cpu().numpy()
    verts_p2 = verts_p2[0].cpu().numpy()
    B, Nv, _ = verts_p1.shape
    return verts_p1, verts_p2, faces, Nv

def get_frame_penetration(v1, v2, faces, Nv, debug=False):

    mesh1 = trimesh.Trimesh(vertices=v1, faces=faces, process=False)
    mesh2 = trimesh.Trimesh(vertices=v2, faces=faces, process=False)
    # assert mesh1.is_watertight and mesh2.is_watertight, "mesh is not watertight"
    # compute penetration
    collision = 0
    mean_penet = 0
    cont_in_mesh1 = mesh1.contains(v2)
    cont_in_mesh2 = mesh2.contains(v1)
    penet1 = np.sum(cont_in_mesh1) / Nv
    penet2 = np.sum(cont_in_mesh2) / Nv
    if penet1 > 0.0 or penet2 > 0.0:
        collision = 1
        mean_penet = (penet1 + penet2) / 2

    if debug:
        op = "inspect_out/metrics/penet"
        save_trimesh(v1, faces, f"{op}/mesh1_{mean_penet:.2f}.ply")
        save_trimesh(v2, faces, f"{op}/mesh2_{mean_penet:.2f}.ply")

    return collision, mean_penet


def compute_penetration(
        smpl_list,
        verbose=False,
        debug=False,
         ):

    verts_p1, verts_p2, faces, Nv = read_verts(smpl_list)
    len_pred = len(verts_p1)
    seq_len = len_pred
    if debug:
        seq_len = 2
    coll_pred_all = []
    coll_gt_all = []
    mean_penet_all = []
    for n, (v1, v2) in enumerate(tqdm(zip(verts_p1[:seq_len], verts_p2[:seq_len]), total=len(verts_p1),
                                      disable=True)):
        coll, mean_penet = get_frame_penetration(v1, v2, faces, Nv)
        coll_pred_all.append(coll)
        mean_penet_all.append(mean_penet)

    coll_pred_all = np.array(coll_pred_all)
    mean_penet_all = np.array(mean_penet_all)

    penet_pred = mean_penet_all[coll_pred_all.astype(bool)].mean()
    if np.isnan(penet_pred):
        penet_pred = 0.0

    if verbose:
        print(f"Mean penetration: {penet_pred:.3f}")
        print(f"Collision frames % : {coll_pred_all.sum()} / {seq_len}")
        print(f"Collision frames GT % : {coll_gt_all.sum()} / {seq_len}")

    penet_GT = 0
    return penet_pred, penet_GT


