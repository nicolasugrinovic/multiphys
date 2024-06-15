import numpy as np
from utils.misc import read_pickle
from utils.video import video_to_images
from utils.video import run_openpose
from pathlib import Path
import os

import numpy as np
from utils.misc import read_pickle
from utils.misc import read_json
from utils.misc import write_pickle
from utils.misc import read_image_PIL
from utils.misc import plot
from utils.misc import plot_joints_cv2
import joblib
from pathlib import Path
import cv2
from utils.smpl import smpl_to_verts
from utils.misc import save_trimesh
from scipy.spatial.transform import Rotation
import os.path as osp
import json
import torch

def prox_to_humor_fmt(seq_name="N0Sofa_00145_01"):
    print(seq_name)
    path = Path(f"/home/nugrinovic/code/CVPR_2024/EmbodiedPose/data/prox/PROXD/{seq_name}")
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

    pass


def get_openpose(flipped=False, seq_name="N0Sofa_00145_01"):
    img_folder = Path(f"data/prox/recordings/{seq_name}")
    # img_process_folder folder where images to be detected are stored
    fname = "Color_flipped" if flipped else "Color"
    img_process_folder = img_folder / Path(f"{fname}")
    # visu video output
    vid_process_folder = img_folder / Path("video")
    op_out_path = img_folder / Path("detections")
    openpose_path = "./external/openpose"

    img_process_folder.mkdir(exist_ok=True, parents=True)
    vid_process_folder.mkdir(exist_ok=True, parents=True)
    op_out_path.mkdir(exist_ok=True, parents=True)

    op_frames_out = os.path.join(img_folder, 'op_frames')
    run_openpose(openpose_path, str(img_process_folder),
                 str(op_out_path),
                 img_out=op_frames_out,
                 video_out=os.path.join(str(vid_process_folder), 'op_keypoints_overlay.mp4'))



def flip_prox_imgs(seq_name="N0Sofa_00145_01"):
    from utils.misc import read_image_PIL
    from utils.misc import save_img
    from pathlib import Path
    import cv2
    from tqdm import tqdm

    image_folder = Path(f"/home/nugrinovic/code/CVPR_2024/EmbodiedPose/data/prox/recordings/{seq_name}/Color")
    out_folder = Path(f"/home/nugrinovic/code/CVPR_2024/EmbodiedPose/data/prox/recordings/{seq_name}/Color_flipped")
    # data = read_pickle(path)
    img_files = sorted(list(image_folder.glob("*.jpg")))
    for idx, imgp in enumerate(tqdm(img_files)):
        # idx = 0
        imgp = img_files[idx]
        im_name = imgp.name
        img = read_image_PIL(imgp)
        # plot(img)
        img_v = cv2.flip(img, 1)
        # plot(img_v)
        outp = out_folder / Path(f"{im_name}")
        outp.parent.mkdir(exist_ok=True, parents=True)
        save_img(outp, img_v)

def save_op_to_pkl(seq_name="N0Sofa_00145_01"):

    op_det_path = Path(f"/home/nugrinovic/code/CVPR_2024/EmbodiedPose/data/prox/recordings/{seq_name}/detections")
    files = sorted(list(op_det_path.glob("*.json")))

    all_op_kpts = []
    for file in files:
        op_data = read_json(file)["people"][0]
        op_jts = np.array(op_data["pose_keypoints_2d"]).reshape(-1, 3)
        # plot_joints_cv2(img, op_jts[None, :, :2], show=True)
        all_op_kpts.append(op_jts)

    all_op_kpts = np.array(all_op_kpts) # (2104, 25, 3)
    out_dict = {
        "joints2d" : all_op_kpts,
    }
    write_pickle(out_dict, op_det_path / Path("../all_op_kpts.pkl"))

    pass


def camera_rotations(seq_name="MPH1Library_00034_01"):


    prox_base = '/home/nugrinovic/code/CVPR_2024/EmbodiedPose/data/prox/qualitative/'

    with open(osp.join(prox_base, f'cam2world/{seq_name[:-9]}.json'), 'r') as f:
        camera_pose = np.array(json.load(f)).astype(np.float32)
        R = camera_pose[:3, :3]
        t = camera_pose[:3, 3]

    np.set_printoptions(precision=3, suppress=True, linewidth=100)

    ### first transform the matrix to euler angles
    R_inv = np.linalg.inv(R)
    r = Rotation.from_matrix(R_inv)
    angles = r.as_euler("xyz", degrees=True)

    #### Modify the angles
    print(f"anges: {angles}")
    print(-t)
    # angles[0] += 5
    #
    # #### Then transform the new angles to rotation matrix again
    # new_r = Rotation.from_euler("xyz", angles, degrees=True)
    # new_rotation_matrix = new_r.as_matrix()
    # diff = np.abs(new_rotation_matrix - R)
    # diff.mean()

def prox_gt_3d_to_2d_kpts(seq_name="MPH11_00034_01"):
    from utils.body_model import pose_to_vertices as pose_to_vertices_
    import smplx
    import trimesh
    from functools import partial
    from utils.misc import save_trimesh
    from utils.misc import save_pointcloud
    from utils.misc import read_npy
    from tqdm import tqdm
    local_bm = smplx.create("data", 'smpl', use_pca=False, batch_size=1)#.cuda()

    np.set_printoptions(precision=3, suppress=True, linewidth=100)

    alignment_base = "/home/nugrinovic/code/CVPR_2024/EmbodiedPose/data/prox/qualitative/alignment"
    proxd_base = "/home/nugrinovic/code/CVPR_2024/EmbodiedPose/data/prox/PROXD"
    prox_base = '/home/nugrinovic/code/CVPR_2024/EmbodiedPose/data/prox/qualitative/'

    with open(osp.join(prox_base, f'cam2world/{seq_name[:-9]}.json'), 'r') as f:
        camera_pose = np.array(json.load(f)).astype(np.float32)
        cwR = camera_pose[:3, :3]
        cwt = camera_pose[:3, 3]

    cam_intrinsics = read_json("/home/nugrinovic/code/CVPR_2024/EmbodiedPose/data/prox/qualitative/calibration/Color.json")
    center = np.array(cam_intrinsics["c"])
    focal = np.array(cam_intrinsics["f"])
    R = np.array(cam_intrinsics["R"])
    T = np.array(cam_intrinsics["T"])
    camera_mtx = np.array(cam_intrinsics["camera_mtx"])
    view_mtx = np.array(cam_intrinsics["view_mtx"])

    align_data = read_npy(f"{alignment_base}/{seq_name[:-9]}.npz")
    aR = align_data["R"]
    at = align_data["t"]

    prox_smpl_path = Path(f'{proxd_base}/{seq_name}/results')
    smpl_files = sorted(list(prox_smpl_path.glob("*/*.pkl")))

    image_folder = Path(f"/home/nugrinovic/code/CVPR_2024/EmbodiedPose/data/prox/recordings/{seq_name}/Color")
    # image_folder = Path(f"/home/nugrinovic/code/CVPR_2024/EmbodiedPose/data/prox/recordings/{seq_name}/Color_flipped")

    for file in smpl_files:
        fname = file.parts[-2]
        img_file = image_folder / Path(f"{fname}.jpg")
        img = read_image_PIL(img_file)
        img_hflip = cv2.flip(img, 1)
        h, w, _ = img.shape
        cx, cy = w // 2, h // 2
        plot(img)
        # body model is
        # {'global_orient': full_pose[:, :3],
        #  'body_pose': full_pose[:, 3:72],
        #  'transl': full_pose[:, 72:],
        #  }
        with torch.no_grad():
            smpl_data = read_pickle(file)
            for k, v in smpl_data.items():
                if k=="body_pose":
                    # (1, 63)
                    smpl_pose = torch.zeros([1, 69])
                    smpl_pose[:, :63] = torch.from_numpy(v)
                    v = smpl_pose
                    smpl_data[k] = v.float()#.cuda()
                else:
                    smpl_data[k] = torch.tensor(v).float()#.cuda()
        output = local_bm(**smpl_data, return_full_pose=True)
        out_path = Path("/home/nugrinovic/code/CVPR_2024/EmbodiedPose/inspect_out/prox/gt_jts")
        smpl_jts = output.joints.cpu().numpy()
        transl = smpl_data["transl"].cpu().numpy()



        frame_name = fname.split("__")[0]
        save_pointcloud(smpl_jts[0], out_path / Path(f"{frame_name}_smpl_jts.ply"))

        # smpl_jts = smpl_jts - smpl_jts[:, 0] + transl[None]
        # save_pointcloud(smpl_jts_root_rel[0, :24], out_path / Path(f"{frame_name}_smpl_jts_root_rel.ply"))

        # # cam to world
        # smpl_jts_w = np.einsum('bij,bkj->bki', cwR[None], smpl_jts) + cwt[None]
        # save_pointcloud(smpl_jts_w[0], out_path / Path(f"{frame_name}_smpl_jts_w.ply"))
        # # align
        # smpl_jts_aligned = np.einsum('bij,bkj->bki', aR[None], smpl_jts_w) + at[None]
        # save_pointcloud(smpl_jts_aligned[0], out_path / Path(f"{frame_name}_aligned.ply"))
        # # align to camera
        # wcR = np.linalg.inv(cwR)
        # smpl_jts_aligned_cam = np.einsum('bij,bkj->bki', wcR[None], smpl_jts_aligned - cwt[None])
        # save_pointcloud(smpl_jts_aligned_cam[0], out_path / Path(f"{frame_name}_smpl_jts_aligned_cam.ply"))
        # cam = cam_intrinsics
        # j2d_cv2 = cv2.projectPoints(smpl_jts_aligned_cam, np.asarray(cam['R']), np.asarray(cam['T']), np.asarray(cam['camera_mtx']),
        #                   np.asarray(cam['k']))[0].squeeze()
        # img_vflip = cv2.flip(img, 0)
        # plot_joints_cv2(img_vflip, j2d_cv2[None], show=True)

        K = torch.zeros([1, 3, 3])
        K[:, 0, 0] = focal[0]
        K[:, 1, 1] = focal[1]
        K[:, 2, 2] = 1.
        K[:, :-1, -1] = torch.from_numpy(center)

        # Apply perspective distortion
        smpl_jts = torch.from_numpy(smpl_jts)
        projected_points = smpl_jts / smpl_jts[:, :, -1].unsqueeze(-1)
        # Apply camera intrinsics
        # projected_points = torch.einsum('bij,bkj->bki', K, projected_points)
        projected_points = (K[0] @ projected_points[0].T).T
        # save_pointcloud(projected_points[0], out_path / Path(f"{frame_name}_projected.ply"))
        # plot_joints_cv2(img_hflip, projected_points[:, :, :2], show=True)
        plot_joints_cv2(img_hflip, projected_points[None, :, :2], show=True)

        cam = cam_intrinsics
        j2d_cv2 = cv2.projectPoints(smpl_jts.numpy(), np.asarray(cam['R']), np.asarray(cam['T']), np.asarray(cam['camera_mtx']),
                          np.asarray(cam['k']))[0].squeeze()
        img_vflip = cv2.flip(img, 0)
        plot_joints_cv2(img_vflip, j2d_cv2[None], show=True)
        



    pass

def prox_smpl_3d_to_2d_op_order(seq_name):
    import smplx
    from tqdm import tqdm

    smpl24_to_op_map = [24, 24, 17, 19, 21, 16, 18, 20, 20, 2, 5, 8, 1, 4, 7]

    local_bm = smplx.create("data", 'smpl', use_pca=False, batch_size=1)#.cuda()
    np.set_printoptions(precision=3, suppress=True, linewidth=100)

    proxd_base = "/home/nugrinovic/code/CVPR_2024/EmbodiedPose/data/prox/PROXD"
    cam_intrinsics = read_json("/home/nugrinovic/code/CVPR_2024/EmbodiedPose/data/prox/qualitative/calibration/Color.json")
    center = np.array(cam_intrinsics["c"])
    focal = np.array(cam_intrinsics["f"])

    prox_smpl_path = Path(f'{proxd_base}/{seq_name}/results')
    smpl_files = sorted(list(prox_smpl_path.glob("*/*.pkl")))
    joints_2d = []
    for file in tqdm(smpl_files):

        # cx, cy = w // 2, h // 2
        # plot(img)
        with torch.no_grad():
            smpl_data = read_pickle(file)
            for k, v in smpl_data.items():
                if k=="body_pose":
                    # (1, 63)
                    smpl_pose = torch.zeros([1, 69])
                    smpl_pose[:, :63] = torch.from_numpy(v)
                    v = smpl_pose
                    smpl_data[k] = v.float()#.cuda()
                else:
                    smpl_data[k] = torch.tensor(v).float()#.cuda()
        output = local_bm(**smpl_data, return_full_pose=True)
        smpl_jts = output.joints.cpu().numpy()

        # fname = file.parts[-2]
        # transl = smpl_data["transl"].cpu().numpy()
        # out_path = Path("/home/nugrinovic/code/CVPR_2024/EmbodiedPose/inspect_out/prox/gt_jts")
        # frame_name = fname.split("__")[0]
        # save_pointcloud(smpl_jts[0], out_path / Path(f"{frame_name}_smpl_jts.ply"))

        # cam = cam_intrinsics
        # j2d_cv2 = cv2.projectPoints(smpl_jts, np.asarray(cam['R']), np.asarray(cam['T']), np.asarray(cam['camera_mtx']),
        #                   np.asarray(cam['k']))[0].squeeze()

        K = torch.zeros([1, 3, 3])
        K[:, 0, 0] = focal[0]
        K[:, 1, 1] = focal[1]
        K[:, 2, 2] = 1.
        K[:, :-1, -1] = torch.from_numpy(center)

        # Apply perspective distortion
        smpl_jts = torch.from_numpy(smpl_jts)
        projected_points = smpl_jts / smpl_jts[:, :, -1].unsqueeze(-1)
        # Apply camera intrinsics
        projected_points = (K[0] @ projected_points[0].T).T

        projected_points = projected_points[None]
        op_kpts = projected_points[:, smpl24_to_op_map]
        op_kpts[:, 1] = (op_kpts[:, 2] + op_kpts[:, 5]) / 2
        op_kpts[:, 8] = (op_kpts[:, 9] + op_kpts[:, 12]) / 2
        joints_2d.append(op_kpts)

        if 0:
            image_folder = Path(f"/home/nugrinovic/code/CVPR_2024/EmbodiedPose/data/prox/recordings/{seq_name}/Color")
            fname = file.parts[-2]
            img_file = image_folder / Path(f"{fname}.jpg")
            img = read_image_PIL(img_file)
            h, w, _ = img.shape
            img_hflip = cv2.flip(img, 1)
            img_vflip = cv2.flip(img, 0)
            plot_joints_cv2(img_hflip, op_kpts[:, :, :2], show=True, with_text=True)

    joints_2d = np.concatenate(joints_2d, axis=0)
    out_path = Path(f"/home/nugrinovic/code/CVPR_2024/EmbodiedPose/data/prox_embodied/{seq_name}/smpl_jts2d.pkl")
    out_path.parent.mkdir(exist_ok=True, parents=True)
    write_pickle(joints_2d, out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    # prox_to_humor_fmt()
    # get_openpose()
    # flip_prox_imgs()
    # get_openpose(flipped=True)
    # save_op_to_pkl()

    # flip_prox_imgs(seq_name="MPH1Library_00034_01")
    # not using the next functions because DEKR works better
    # get_openpose(flipped=True, seq_name="MPH1Library_00034_01")
    # save_op_to_pkl(seq_name="N0Sofa_00145_01")

    # prox_to_humor_fmt(seq_name="MPH1Library_00034_01")
    # this one is for defining the camera in the simu, but actually need more work to make it work
    # camera_rotations(seq_name="MPH1Library_00034_01")

    # prox_gt_3d_to_2d_kpts()
    # prox_gt_3d_to_2d_kpts("MPH112_00034_01")
    prox_smpl_3d_to_2d_op_order("MPH11_00034_01")
