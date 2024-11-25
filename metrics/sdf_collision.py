import numpy as np
# from sdf import SDFLoss
from metrics.sdf_loss import SDFLoss
import smplx
import torch
from metrics.collision import read_verts
from tqdm import tqdm
np.set_printoptions(precision=4)
np.set_printoptions(precision=3, suppress=True, linewidth=100)


def compute_collision_sdf(smpl_list, key, args, n, verbose=False, threshold=0.0):
    """
    Compute inter-person penetration using SDF. This one is used to compute the final results for the
    MultiPhys paper.
    """
    verts_p1, verts_p2, _, Nv = read_verts(smpl_list) # each verts_p is (150, 6890, 3) --> [B, Nv, 3]
    local_bm = smplx.create("data", 'smpl', use_pca=False, batch_size=1)
    faces = local_bm.faces

    # faces = faces.astype(np.int32)
    # faces = torch.tensor(faces).cuda()
    sdf_loss_fnc = SDFLoss(faces, debugging=False, robustifier=None)
    sdf_loss_fnc.faces = sdf_loss_fnc.faces.cuda()
    v1 = torch.tensor(verts_p1).cuda() # [B, Nv, 3]
    v2 = torch.tensor(verts_p2).cuda() # [B Nv, 3]
    pred_vertices = torch.stack([v1, v2], dim=0) # [2, B Nv, 3]
    pred_vertices = pred_vertices.permute(1, 0, 2, 3) # [B, 2, Nv, 3]
    # pred_vertices contains the verts for each person for all frames
    loss = []
    for iter, pred_v in enumerate(tqdm(pred_vertices, disable=not verbose)):
        # this computes the sdf loss per frame
        # pred_v is [2, Nv, 3]
        translation = torch.zeros([2, 3]).cuda() # pred_v is for one frame
        # sdf_loss is a scalar here
        sdf_loss = sdf_loss_fnc(pred_v, translation, iter=iter, threshold=threshold)
        # loss_sdf = sdf_loss.sum() if use_sdf else sdf_loss.sum().detach() * 1e-4
        loss.append(sdf_loss.item())

    loss = np.array(loss)
    loss_val = loss.sum()

    debug = False
    if debug:
        max_idx = np.argmax(loss)
        max_loss = loss.max()
        # loss_val_i = loss[max_idx]
        loss_tup = [(i, c) for i, c in enumerate(loss)]


        from utils.smpl import save_mesh
        # max_idx=174
        op = f"inspect_out/compute_metrics/sdf/{args.data_name}-{args.model_type}/{key}"
        save_mesh(pred_vertices[max_idx], faces=faces, out_path=f"{op}/meshes_{n}_{max_idx:04d}.ply")

    return loss_val



if __name__ == "__main__":
    local_bm = smplx.create("data", 'smpl', use_pca=False, batch_size=1)
    faces = local_bm.faces
    sdf_loss = SDFLoss(faces, debugging=False, robustifier=None)

