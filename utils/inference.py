import os
import os.path as osp
import sys

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.append(os.getcwd())
import numpy as np
from uhc.smpllib.smpl_eval import compute_metrics
from pathlib import Path
from utils.misc import save_mesh


def get_datetime():
    import datetime
    ct = datetime.datetime.now()
    data_time = ct.strftime("%Y-%m-%d_%H-%M-%S")
    return data_time




def save_meshes(pred_verts, gt_verts, faces, output_dir):
    mesh_folder = osp.join(output_dir, f"meshes")
    Path(mesh_folder).mkdir(parents=True, exist_ok=True)
    pred_verts = np.stack(pred_verts, axis=0)
    pred_verts = np.transpose(pred_verts, (1, 0, 2, 3))
    for ni, p_verts in enumerate(pred_verts):
        save_mesh(p_verts, faces=faces, out_path=f"{mesh_folder}/verts_{ni:04d}.ply")

    mesh_folder_gt = osp.join(output_dir, f"meshes_GT")
    Path(mesh_folder_gt).mkdir(parents=True, exist_ok=True)
    gt_verts = np.stack(gt_verts, axis=0)
    gt_verts = np.transpose(gt_verts, (1, 0, 2, 3))
    for ni, p_verts in enumerate(gt_verts):
        save_mesh(p_verts, faces=faces, out_path=f"{mesh_folder_gt}/verts_{ni:04d}.ply")
    print("**Meshes saved at:")
    print(mesh_folder)
    print("********************")


def print_metrics(eval_res_n):
    print_str = "\t".join([f"{k}: {np.mean(v):.3f}" for k, v in eval_res_n.items() if not k in [
        "gt",
        "pred",
        "pred_jpos",
        "gt_jpos",
        "reward",
        "gt_vertices",
        "pred_vertices",
        "gt_joints",
        "pred_joints",
        "action",
        "vf_world",
    ] and (not isinstance(v, np.ndarray))])
    print(print_str)

    # do evaluation
    metric_res = compute_metrics(eval_res_n, None)
    metric_res = {k: np.mean(v) for k, v in metric_res.items()}
    print_str = " \t".join([f"{k}: {v:.3f}" for k, v in metric_res.items()])
    print_str += print_str + "\n"

    print_str_eval = " \t ".join([f"{k}" for k, v in metric_res.items()])  # keys
    print_str_eval += "\n "
    print_str_eval += " \t ".join([f"{v:.3f}" for k, v in metric_res.items()])  # values
    # save the results
    result = eval_res_n.copy()
    result.update(metric_res)
    # results_all.append(result)
    # print_eval_all += print_str_eval + "\n"
    return result, print_str_eval