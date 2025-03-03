import numpy as np
from utils.misc import read_pickle
from utils.misc import save_trimesh
from utils.misc import plot
from utils.misc import read_image_PIL
from utils.misc import plot_joints_cv2
from utils.misc import save_pointcloud
from utils.misc import write_pickle
import smplx
from tqdm import tqdm

from utils.smpl import smpl_to_verts
import smplx
import torch

np.set_printoptions(precision=6, suppress=True, linewidth=100)
from pathlib import Path

# def project_3d_to_2d_simple(points3d, focal, center):
#     """ this is for wo_distortion only """
#     xx = points3d[:, :, :2] / points3d[:, :, 2:3]
#     proj = focal[None] * xx + center[None]
#     return proj


def rigid_transform_3D(A, B):
    assert A.shape == B.shape
    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")
    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")
    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)
    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)
    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B
    H = Am @ np.transpose(Bm)
    # sanity check
    # if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))
    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    t = -R @ centroid_A + centroid_B
    return R, t


def get_cam2world(seq_data):
    # {'global_orient': full_pose[:, :3],
    #  'body_pose': full_pose[:, 3:72],
    #  'transl': full_pose[:, 72:],
    #  }

    # img_path = Path("/home/nugrinovic/code/CVPR_2024/EmbodiedPose/data/chi3d/openpose/images")
    # img_files = sorted(list(img_path.glob("*.png")))
    # img_f = img_files[0]

    # file_path = '/home/nugrinovic/code/CVPR_2024/EmbodiedPose/data/chi3d/chi3d_s02_embodied.pkl'
    # data = read_pickle(file_path)
    # keys ['trans', 'root_orient', 'pose_body', 'betas', 'cam_50591643', 'cam_58860488',
    # 'cam_60457274', 'cam_65906101', 'joints3d', 'joints2d'])
    # seq_data = data[seq_name]
    cam = seq_data['cam_50591643']

    ori = seq_data['root_orient']
    body = seq_data['pose_body']
    pose = np.zeros([1, 72])
    pose[:, :3] = ori[0, None]
    pose[:, 3:66] = body[0, None]
    trans = seq_data['trans'][0, None]
    # verts, faces = smpl_to_verts(pose, trans)
    # save_trimesh(verts[0, 0], faces, "inspect_out/prox/meshes/process/pose_noisy_rot.ply")
    inspect_path = "inspect_out/chi3d/meshes/process/"
    # save_trimesh(verts[0, 0], faces, "inspect_out/chi3d/meshes/process/vert_gt.ply")
    local_bm = smplx.create("data", 'smpl', use_pca=False, batch_size=1)  # .cuda()
    smpl_data =  {'global_orient': torch.from_numpy(pose[:, :3]).float(),
                 'body_pose': torch.from_numpy(pose[:, 3:72]).float(),
                 'transl': trans.float(),
                 }

    with torch.no_grad():
        output = local_bm(**smpl_data, return_full_pose=True)
        smpl_jts = output.joints[:, :24]
    # img = read_image_PIL(img_f)
    # plot(img)
    # is this camera to world?
    R = np.array(cam['R'])
    T = np.array(cam['T'])
    K = cam['K']
    
    #######################################
    # smpl_jts is ([1, 24, 3]), R is 3x3
    # R_inv = np.linalg.inv(R)
    # R_trans = R.transpose(1, 0)
    # R_inv == R_trans --> it is the same! although == does not say so
    j3d_in_camera = np.matmul(smpl_jts.numpy() - T[None], R[None].transpose(0, 2, 1)) # (219, 25, 3)

    # smpl_trans = smpl_jts[0, 0]
    # j3d_cam_trans = j3d_in_camera[0, 0]
    # j3d_in_camera_smpl = j3d_in_camera - j3d_cam_trans[None, None] + smpl_trans[None, None].numpy()
    # save_pointcloud(j3d_in_camera_smpl[0], Path(f"{inspect_path}/j3d_in_camera_smpl.ply"))

    smpl_jts_np = smpl_jts.numpy()
    # A, B se obtiene transf de A a B
    ret_R, ret_t = rigid_transform_3D(j3d_in_camera[0].T, smpl_jts_np[0].T)

    # smpl_jts_transf = (ret_R @ j3d_in_camera[0].T).T + ret_t.T
    # save_pointcloud(smpl_jts_transf, Path(f"{inspect_path}/smpl_jts_transf.ply"))
    # save_pointcloud(smpl_jts[0], Path(f"{inspect_path}/smpl_jts.ply"))
    # diff = np.abs(smpl_jts_transf-smpl_jts_np[0]).mean()

    cwR = ret_R
    cwT = ret_t
    # wcR = np.linalg.inv(cwR)
    # smpl_jts_cam = (wcR @ (smpl_jts_np[0] - cwT.T).T).T
    # save_pointcloud(smpl_jts_cam, Path(f"{inspect_path}/smpl_jts_cam.ply"))

    # center = np.array(K["c"])
    # focal = np.array(K["f"])
    # kpts2d = project_3d_to_2d_simple(j3d_in_camera, center, focal)
    # xx = j3d_in_camera[:, :, :2] / j3d_in_camera[:, :, 2:3]
    # kpts2d = focal[None] * xx[0] + center[None]
    # plot_joints_cv2(img, kpts2d[None], show=True)

    # smpl_jts_cam = smpl_jts_cam[None]
    # xx = smpl_jts_cam[:, :, :2] / smpl_jts_cam[:, :, 2:3]
    # kpts2d = focal[None] * xx[0] + center[None]
    # plot_joints_cv2(img, kpts2d[None], show=True)

    # seq_data = data[seq_name]
    cwT = cwT.T
    # seq_data["cam2wold"] = {"R": cwR.tolist(), "T": cwT.tolist(), "K": K}
    # cwT == R.T
    # out_path = '/home/nugrinovic/code/CVPR_2024/EmbodiedPose/data/chi3d/chi3d_s02_embodied_cam2w.pkl'
    # write_pickle(data, out_path)

    return {"R": cwR.tolist(), "T": cwT.tolist(), "K": K}



if __name__ == "__main__":
    get_cam2world()
    # rigid_transform()
