import json
import numpy as np
import cv2
import matplotlib.pyplot as plt


def read_video(vid_path):
    frames = []
    cap = cv2.VideoCapture(vid_path)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) )
    cap.release()
    frames = np.array(frames)
    return frames

def read_cam_params(cam_path):
    with open(cam_path) as f:
        cam_params = json.load(f)
        for key1 in cam_params:
            for key2 in cam_params[key1]:
                cam_params[key1][key2] = np.array(cam_params[key1][key2]) 
    return cam_params

def read_data(data_root, dataset_name, subset, subj_name, action_name, camera_name, subject="w_markers"):
    if subject == 'wo_markers':
        assert(dataset_name == 'chi3d')
    vid_path = '%s/%s/%s/%s/videos/%s/%s.mp4' % (data_root, dataset_name, subset, subj_name, camera_name, action_name)
    cam_path = '%s/%s/%s/%s/camera_parameters/%s/%s.json' % (data_root, dataset_name, subset, subj_name, camera_name, action_name)
    j3d_path = '%s/%s/%s/%s/joints3d_25/%s.json' % (data_root, dataset_name, subset, subj_name, action_name)
    gpp_path = '%s/%s/%s/%s/gpp/%s.json' % (data_root, dataset_name, subset, subj_name, action_name)
    smplx_path = '%s/%s/%s/%s/smplx/%s.json' % (data_root, dataset_name, subset, subj_name, action_name)

    cam_params = read_cam_params(cam_path)

    with open(j3d_path) as f:
        j3ds = np.array(json.load(f)['joints3d_25'])
    seq_len = j3ds.shape[-3]
    with open(gpp_path) as f:
        gpps = json.load(f)
    with open(smplx_path) as f:
        smplx_params = json.load(f)
    frames = read_video(vid_path)[:seq_len]
    
    dataset_to_ann_type = {'chi3d': 'interaction_contact_signature', 
                           'fit3d': 'rep_ann', 
                           'humansc3d': 'self_contact_signature'}
    ann_type = dataset_to_ann_type[dataset_name]
    annotations = None
    if ann_type:
        ann_path = '%s/%s/%s/%s/%s.json' % (data_root, dataset_name, subset, subj_name, ann_type)
        with open(ann_path) as f:
            annotations = json.load(f)
    
    if dataset_name == 'chi3d': # 2 people in each frame
        subj_id = 0 if subject == "w_markers" else 1
        j3ds = j3ds[subj_id, ...]
        for key in gpps:
            gpps[key] = gpps[key][subj_id]
        for key in smplx_params:
            smplx_params[key] = smplx_params[key][subj_id]
        
    
    return frames, j3ds, cam_params, gpps, smplx_params, annotations


def project_3d_to_2d(points3d, intrinsics, intrinsics_type):
    if intrinsics_type == 'w_distortion':
        p = intrinsics['p'][:, [1, 0]]
        x = points3d[:, :2] / points3d[:, 2:3]
        r2 = np.sum(x**2, axis=1)
        radial = 1 + np.transpose(np.matmul(intrinsics['k'], np.array([r2, r2**2, r2**3])))
        tan = np.matmul(x, np.transpose(p))
        xx = x*(tan + radial) + r2[:, np.newaxis] * p
        proj = intrinsics['f'] * xx + intrinsics['c']
    elif intrinsics_type == 'wo_distortion':
        xx = points3d[:, :2] / points3d[:, 2:3]
        proj = intrinsics['f'] * xx + intrinsics['c']
    return proj

def project_3d_to_2d_simple(points3d, intrinsics):
    """ this is for wo_distortion only """
    xx = points3d[:, :, :2] / points3d[:, :, 2:3]
    proj = intrinsics['f'][None] * xx + intrinsics['c'][None]
    return proj

def plot_over_image(frame, points_2d=np.array([]), with_ids=True, with_limbs=True, path_to_write=None):
    num_points = points_2d.shape[0]
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(frame)
    if points_2d.shape[0]:
        ax.plot(points_2d[:, 0], points_2d[:, 1], 'x', markeredgewidth=10, color='white')
        if with_ids:
            for i in range(num_points):
                ax.text(points_2d[i, 0], points_2d[i, 1], str(i), color='red', fontsize=20)
        if with_limbs:
            limbs = [[10, 9], [9, 8], [8, 11], [8, 14], [11, 12], [14, 15], [12, 13], [15, 16],
                    [8, 7], [7, 0], [0, 1], [0, 4], [1, 2], [4, 5], [2, 3], [5, 6],
                    [13, 21], [13, 22], [16, 23], [16, 24], [3, 17], [3, 18], [6, 19], [6, 20]]
            for limb in limbs:
                if limb[0] < num_points and limb[1] < num_points:
                    ax.plot([points_2d[limb[0], 0], points_2d[limb[1], 0]], 
                            [points_2d[limb[0], 1], points_2d[limb[1], 1]],
                            linewidth=12.0)
            
    plt.axis('off')
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    if path_to_write:
        plt.ioff()
        plt.savefig(path_to_write, pad_inches = 0, bbox_inches='tight')

