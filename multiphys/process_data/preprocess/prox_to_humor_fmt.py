from pathlib import Path
import numpy as np
from utils.misc import read_pickle
from utils.misc import write_pickle
import argparse

def prox_to_humor_fmt(
        # seq_name="N0Sofa_00145_01",
        path=f"/home/nugrinovic/code/CVPR_2024/EmbodiedPose/data/prox/PROXD/N0Sofa_00145_01"
        ):

    path = Path(path)
    print(path.stem)
    seq_name = path.stem
    data_files = sorted(list(path.glob("*/*/*.pkl")))
    # datap = Path("results/s001_frame_00001__00.00.00.026/000.pkl")
    datap = data_files[0]
    # data --> keys: (['camera_rotation', 'camera_translation', 'betas', 'global_orient', 'transl', 'left_hand_pose',
    # 'right_hand_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'expression', 'pose_embedding', 'body_pose'])
    data = read_pickle(path / datap)
    transl = data["transl"]
    global_orient = data["global_orient"]
    body_pose = data["body_pose"]
    betas = data["betas"]
    # camera_rotation = data["camera_rotation"]
    # camera_translation = data["camera_translation"]
    # NOTE: Humor dict keys used are trans, pose_body, root_orient
    humor = {
        "trans": transl,
        "pose_body": body_pose,
        "root_orient": global_orient,
        "betas": betas,
    }

    pose = humor['pose_body']
    ori = humor['root_orient']
    trans = humor['trans']
    pose_72 = np.zeros([1, 72])
    pose_72[:, :3] = ori
    pose_72[:, 3:66] = pose
    # the input to smpl_to_verts is a (1, 72) vector of pose parameters
    # verts, faces = smpl_to_verts(pose_72, trans)
    # save_trimesh(verts[0, 0], faces, "inspect_out/prox/meshes/smpl_gt.ply")
    out_path = Path(f"/home/nugrinovic/code/CVPR_2024/EmbodiedPose/data/prox_embodied/{seq_name}")
    out_path.mkdir(exist_ok=True, parents=True)
    write_pickle(humor, out_path / Path("prox_humor_fmt.pkl"))


if __name__ == "__main__":
    from tqdm import tqdm
    # parser = argparse.ArgumentParser(description='convert format')
    # parser.add_argument('--seq_name', type=str, required=True)
    # args = parser.parse_args()

    prox_path = Path("/home/nugrinovic/code/CVPR_2024/EmbodiedPose/data/prox/PROXD")
    seq_folders = sorted(list(prox_path.glob("*")))[1:]
    for seq_path in tqdm(seq_folders):
        prox_to_humor_fmt(seq_path)
