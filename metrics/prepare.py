import numpy as np
import torch
from utils.pyquaternion import Quaternion as Q
from utils.misc import read_pickle
from utils.smpl import smpl_to_verts
from utils.net_utils import get_hostname
from scipy.spatial.transform import Rotation as sRot
from metrics.tools import match_w_hungarian
from tqdm import tqdm
from pathlib import Path
import joblib

T = Q(axis=[0, 0, 1], angle=np.pi).transformation_matrix.astype(np.float32)
Tz180 = torch.from_numpy(T).cuda()

hostname = get_hostname()


def get_emb_paths(data_name, exp_name, slahmr_overwrite=False, get_add=False, sla_root=False):

    ROOT = "/home/nugrinovic/code/CVPR_2024/EmbodiedPose"
    SLA_ROOT = "/home/nugrinovic/code/CVPR_2024/slahmr_release/slahmr"

    RES_ROOT = f"{ROOT}/results/scene+/tcn_voxel_4_5_chi3d_multi_hum/results"
    SAMP_DATA_ROOT = f"{ROOT}/sample_data"
    path = f"{RES_ROOT}/{data_name}/{exp_name}"

    PROC_ROOT = f"{ROOT}/sample_data"

    if slahmr_overwrite:
        sdata_path = f"{SAMP_DATA_ROOT}/{data_name}_slahmr"
    else:
        sdata_path = f"{SAMP_DATA_ROOT}/{data_name}"

    if data_name == 'chi3d':
        SEQS = ["s02", "s03", "s04"]
    elif data_name == 'expi':
        SEQS = ["acro1", "acro2"]
    else:
        SEQS = [data_name]

    if sla_root:
        return path, SEQS, ROOT, RES_ROOT, SLA_ROOT, PROC_ROOT
    if get_add:
        return path, SEQS, ROOT, RES_ROOT, sdata_path
    return path, SEQS, ROOT, RES_ROOT


def get_proc_data(args, PROC_ROOT, subj_name):

    """
    especially for prepare_pred_results_expi.py
    """

    sub_n = f"_{subj_name}" if subj_name != args.data_name else ""
    if args.data_name == 'chi3d':
        p1 = f"{PROC_ROOT}/{args.data_name}_slahmr/{args.data_name}_slahmr{sub_n}_p1_phalpBox_all_slaInit_slaCam.pkl"
        p2 = f"{PROC_ROOT}/{args.data_name}_slahmr/{args.data_name}_slahmr{sub_n}_p2_phalpBox_all_slaInit_slaCam.pkl"
    elif args.data_name == 'hi4d':
        p1 = f"{PROC_ROOT}/{args.data_name}_slahmr/{args.data_name}{sub_n}_p1_phalpBox_all_slaInit_slaCam.pkl"
        p2 = f"{PROC_ROOT}/{args.data_name}_slahmr/{args.data_name}{sub_n}_p2_phalpBox_all_slaInit_slaCam.pkl"
    else:
        p1 = f"{PROC_ROOT}/{args.data_name}/{args.data_name}{sub_n}_p1_phalpBox_all_slaInit_slaCam.pkl"
        p2 = f"{PROC_ROOT}/{args.data_name}/{args.data_name}{sub_n}_p2_phalpBox_all_slaInit_slaCam.pkl"

    print(f"Reading PROC DATA from {p1} | data_name: {args.data_name}")

    proc_data = [joblib.load(p1)]
    keys = list(proc_data[0].keys())
    print(f"\nKEYS from PROC DATA are: {keys}")
    print(f"LEN of PROC DATA is: {len(keys)}\n")
    proc_data.append(joblib.load(p2))
    path = Path(p1)
    # proc_data is a list of len=2 containing dicts with the sequence names as keys
    return proc_data, path


def get_subj_paths(args, RES_ROOT, RES_NORM_ROOT, subj_name):
    """
    especially for prepare_pred_results_expi.py
    """
    path = f"{RES_ROOT}/{args.data_name}/{args.exp_name}/{subj_name}"
    path = Path(path)
    path_norm_op = Path(f"{RES_NORM_ROOT}/{subj_name}")
    return path, path_norm_op

def get_results_latest_path(emb_root, seq_name, res_name="results"):
    embvid_path = f"{emb_root}/{seq_name}"
    embvid_path = Path(embvid_path)
    emb_dir = sorted(embvid_path.glob('*'))[-1]
    embv_p = emb_dir / f"{res_name}.pkl"
    return embv_p

def get_latest_files(path):
    files = sorted((path.glob(f"*")))
    # keep only folders
    files = [f for f in files if f.is_dir()]
    last_files = []
    for file in files:
        pass
        seq_name = file.stem
        f = get_results_latest_path(str(path), seq_name, res_name="results")
        last_files.append(f)
    files = last_files
    return files


def get_all_results(path, remove_subj=False, is_heuristic=False):
    if isinstance(path, str):
        path = Path(path)


    files = get_latest_files(path)
    exp_name = path.parts[-2]

    if is_heuristic:
        # pad heuristics results with original results as heuristics can be a subset of the original results
        orig_exp = exp_name.split("_optim_naive_heuristic")[0]
        new_path = str(path).replace(exp_name, orig_exp)
        ref_files = get_latest_files(Path(new_path))
        files_dict = {c.parts[-3]:c for c in files}
        ref_dict = {c.parts[-3]:c for c in ref_files}
        ref_dict.update(files_dict)
        files = list(ref_dict.values())

    all_results = {}
    for file in tqdm(files, disable=True):
        try:
            data = read_pickle(file)
            if remove_subj and '/expi/' in str(path):
                k, v = list(data.items())[0]
                data = {k[6:]: v}
        except:
            print(f"does not exist {file}! ")
            continue
        all_results.update(data)

    keys = list(all_results.keys())
    print(f"\nKEYS from RESULTS are: {keys}")
    print(f"LEN of RESULTS is: {len(keys)}\n")

    return all_results


def rotate_180z(pred_jpos):
    Tz180 = Q(axis=[0, 0, 1], angle=np.pi).transformation_matrix.astype(np.float32)
    pred_jpos_r = (Tz180[None, :3, :3] @ pred_jpos.transpose(0, 2, 1)).transpose(0, 2, 1)
    return pred_jpos_r

def rotate_z(pred_jpos, ang_rad=np.pi):
    Tz180 = Q(axis=[0, 0, 1], angle=ang_rad).transformation_matrix.astype(np.float32)
    pred_jpos_r = (Tz180[None, :3, :3] @ pred_jpos.transpose(0, 2, 1)).transpose(0, 2, 1)
    return pred_jpos_r

def get_smpl_joints_from_expi(expi_joints):
    vis_mask = np.array([1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0])
    expi_to_smpl_map = [10, 10, 11, 11, 12, 13, 13, 14, 15, 15, 16, 17, 17,
                        17, 17, 17, 4, 5, 6, 7, 8, 9, 9, 9]
    smpl_joints = expi_joints[:, expi_to_smpl_map]
    smpl_joints[:, 0] = (expi_joints[:, 10] + expi_joints[:, 11]) / 2
    smpl_joints = smpl_joints.reshape(-1, 72)
    return smpl_joints, vis_mask

def get_smpl_joints_from_expi_mp(expi_joints):
    vis_mask = np.array([1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]).astype(bool)
    expi_to_smpl_map = [10, 10, 11, 11, 12, 13, 13, 14, 15, 15, 16, 17, 17,
                        17, 17, 17, 4, 5, 6, 7, 8, 9, 9, 9]
    expi_joints = np.stack(expi_joints)
    smpl_joints = expi_joints[:, :, expi_to_smpl_map]
    smpl_joints[:, :, 0] = (expi_joints[:, :, 10] + expi_joints[:, :, 11]) / 2
    smpl_joints = smpl_joints.reshape(2, -1, 72)
    return smpl_joints, vis_mask



def get_cam_from_seq_preds(proc_data):
    try:
        full_R = proc_data['cam']['full_R']
        full_t = proc_data['cam']['full_t']
        K = proc_data['cam']['K']
    except:
        full_R = proc_data['cam'][0]['full_R']
        full_t = proc_data['cam'][0]['full_t']
        K = proc_data['cam'][0]['K']

    T_w2c = proc_data['cam'][0]['T_w2c']

    final_Rt = proc_data['final_Rt']
    final_Rt_inv = proc_data['final_Rt_inv']

    focal = np.array([K[0, 0], K[1, 1]])[None]
    center = np.array([K[0, 2], K[1, 2]])[None]
    cam_dict = {'cam2world': {  'R': full_R,
                                'T': full_t,
                                'K': K,
                                'focal': focal,
                                'center': center,
                                'T_w2c': T_w2c},
                'final_Rt': final_Rt,
                'final_Rt_inv': final_Rt_inv
                }
    return cam_dict

def get_cam_from_preds(proc_data, seq_name):
    try:
        full_R = proc_data[0][seq_name]['cam']['full_R']
        full_t = proc_data[0][seq_name]['cam']['full_t']
    except:
        full_R = proc_data[0][seq_name]['cam'][0]['full_R']
        full_t = proc_data[0][seq_name]['cam'][0]['full_t']
    cam_dict = {'cam2world': {'R': full_R, 'T': full_t}}
    return cam_dict

def parse_cam2world(gt_data_p1, tensor=True, device='cuda'):
    cam2world_gt = gt_data_p1['cam2world']
    R = np.array(cam2world_gt['R'])
    T = np.array(cam2world_gt['T'])
    Rt = np.zeros([4, 4])
    Rt[:3, :3] = R
    Rt[:3, 3] = T
    Rt[3, 3] = 1
    Rt = Rt[None].astype(np.float32)
    if tensor:
        Rt = torch.from_numpy(Rt).to(device)
    return Rt

def parse_seq_gt_data(data_chi3d, data_chi3d_2, seq_name):

    gt_p1 = data_chi3d[seq_name]
    gt_p2 = data_chi3d_2[seq_name]
    gt_chi3d = [gt_p1['joints_3d'], gt_p2['joints_3d']]
    Rt = parse_cam2world(gt_p1)

    return gt_chi3d, Rt


def get_camera_transform(seq, vid_name, return_gnd=False, data_name='chi3d'):
    if "oriong" in hostname:
        SLA_ROOT = "/sailhome/nugrinov/code/CVPR_2024/slahmr_release/slahmr"
    else:
        SLA_ROOT = "/home/nugrinovic/code/CVPR_2024/slahmr_release/slahmr"

    res_dir = f"{SLA_ROOT}/outputs/logs/{data_name}-val/{seq}/{vid_name}-all-shot-0-0-180"
    scene_dict = read_pickle(f"{res_dir}/{vid_name}_scene_dict.pkl")
    Rt_gnd = scene_dict["ground"]
    # this is T_c2w from slahmr
    T_c2w = scene_dict["cameras"]["src_cam"]
    # this is the ground transform from slahmr, inverted as we want to take the poses to xy, z=0
    Rt_gnd_inv = torch.linalg.inv(Rt_gnd).to(T_c2w.device)
    # rotate by -90 deg around x to match the +z up camera from EmbodiedPose
    Txm90 = Q(axis=[1, 0, 0], angle=-np.pi / 2).transformation_matrix.astype(np.float32)
    Txm90 = torch.tensor(Txm90).float().to(T_c2w.device)
    # T_c2w contains cameras for each frame, use this for dynamic camera
    final_Rt = Txm90[None] @ Rt_gnd_inv[None] #@ T_c2w
    scene_dict['final_Rt'] = final_Rt
    if return_gnd:
        Rx90_Rtgnd = Txm90[None] @ Rt_gnd.to(T_c2w.device)
        return scene_dict, Rx90_Rtgnd
    return scene_dict


def rot_pose(pose_aa, final_Rt):
    pose_aa_i = pose_aa.clone()
    ori = torch.tensor(sRot.from_rotvec(pose_aa_i[:, :3]).as_matrix()).to(final_Rt)
    new_root = final_Rt[:, :3, :3] @ ori
    new_root = sRot.from_matrix(new_root.cpu()).as_rotvec()
    pose_aa_i[:, :3] = torch.tensor(new_root)
    return pose_aa_i


def correct_smplx_offset(pose_aa_i, trans_i, betas, final_Rt, get_verts=False):
    (_, joints), _ = smpl_to_verts(pose_aa_i, trans_i, betas=betas, return_joints=True)
    pelvis = joints[0, :, 0, None] # (B, 1, 3)
    trans_i = trans_i[:, None]
    pelvis = pelvis.to(trans_i) - trans_i
    trans_i = (final_Rt[:, :3, :3] @ (trans_i + pelvis).permute(0, 2, 1)).permute(0, 2, 1) + final_Rt[:, None, :3, 3] - pelvis
    trans_i = trans_i[:, 0]
    if get_verts:
        verts, faces = smpl_to_verts(pose_aa_i, trans_i, betas=betas, return_joints=False)
        return trans_i, verts, faces
    return trans_i, None, None


def read_smpl_meshes(data_chi3d_this, seq_len=None, get_joints=False):
    root_orient = data_chi3d_this["root_orient"]
    pose_body = data_chi3d_this["pose_body"]
    trans = data_chi3d_this["trans"]
    betas = data_chi3d_this["betas"]
    if seq_len is None:
        seq_len = root_orient.shape[0]
    pose_aa = torch.cat([root_orient[:seq_len], pose_body[:seq_len], torch.zeros(seq_len, 6).float()], dim=-1)
    out = smpl_to_verts(pose_aa, trans[:seq_len].float(), betas=betas[:1], return_joints=get_joints)
    if get_joints:
        (verts, joints), faces = out
        return verts, joints, faces
    verts, faces = out
    return verts, faces


def parse_smpl_data(res_dict, data_name='expi'):

    root_aa = res_dict['root_orient']
    body_pose_aa = res_dict["pose_body"]
    trans = res_dict["trans"]
    shape = res_dict["betas"][..., :10]

    if isinstance(root_aa, torch.Tensor):
        root_aa = root_aa.cpu()
    else:
        root_aa = torch.tensor(root_aa).float()
    if isinstance(body_pose_aa, torch.Tensor):
        body_pose_aa = body_pose_aa.cpu()
    else:
        body_pose_aa = torch.tensor(body_pose_aa).float()
    if isinstance(trans, torch.Tensor):
        trans = trans.cpu()
    else:
        trans = torch.tensor(trans).float()
    if isinstance(shape, torch.Tensor):
        shape = shape.cpu()
    else:
        shape = torch.tensor(shape).float()

    B = root_aa.shape[0]

    body_dim = body_pose_aa.shape[-1]
    if body_dim==69:
        pose_aa = torch.cat([root_aa, body_pose_aa], dim=-1)
    elif body_dim==63:
        pose_aa = torch.cat([root_aa, body_pose_aa, torch.zeros(B, 6).to(root_aa)], dim=-1)
    else:
        raise ValueError(f"ERROR: Weird body_dim: {body_dim}!!")

    return_dict = {
        'pose_aa': pose_aa,
        'trans': trans,
        'shape': shape,
    }
    return return_dict


def read_slahmr_transform_and_body(seq, vid_name, data_name='chi3d'):
    try:
        res_dict, Rx90_Rtgnd = get_camera_transform(seq, vid_name, return_gnd=True, data_name=data_name)
        # out = get_slahmr_emb2cam_transform(data_name, seq_name[4:], False, seq_num)
        # rotation, translation, focal_length, camera_center, _ = out
    except:
        print(f"skipping {vid_name}, transforms data does not exist! generate '*_scene_dict.pkl' in slahrm")
        print(f"skipping {vid_name}, transforms data does not exist! generate '*_scene_dict.pkl' in slahrm")
        return None, None
    # for now using only the first cam, but later use all
    final_Rt = res_dict['final_Rt']
    jpos_pred_ = res_dict["joints"]
    # SMPL-H joints, not SMPL
    jpos_pred = jpos_pred_[:, :, :24]  # these have one wrong joint, correct this!
    # this corrects the erroneous joint, due to the SMPL-H model
    jpos_pred[:, :, -1] = jpos_pred_[:, :, 37]
    root_aa = res_dict['root_orient'].cpu()
    body_pose_aa = res_dict["pose_body"].cpu()
    B = root_aa.shape[0]
    seq_len = root_aa.shape[1]
    pose_aa = torch.cat([root_aa, body_pose_aa, torch.zeros(B, seq_len, 6).to(root_aa)], dim=-1)
    trans = res_dict["trans"].cpu()
    shape = res_dict["betas"][..., :10].cpu()
    return_dict= {
        'final_Rt': final_Rt,
        'jpos_pred': jpos_pred,
        'pose_aa': pose_aa,
        'trans': trans,
        'shape': shape,
        'Rx90_Rtgnd': Rx90_Rtgnd,

    }
    return return_dict, seq_len



def apply_transform(jpos_pred, final_Rt):
    jpos_pred_world = (final_Rt[:, :3, :3] @ jpos_pred.permute(0, 2, 1)).permute(0, 2, 1) + final_Rt[:, None, :3, 3]
    return jpos_pred_world


def transform_pred_joints(jpos_pred, final_Rt, rot_180z=True):
    j_pred_rot = []
    for p_id in range(len(jpos_pred[:2])):
        # NOTE: for some reason we have to rotate the joints and SMPL 180 over z-axis (check why).
        # transform R|t
        jpos_pred_world = apply_transform(jpos_pred[p_id], final_Rt)
        if rot_180z:
            T = Q(axis=[0, 0, 1], angle=np.pi).transformation_matrix.astype(np.float32)
            Tz180 = torch.from_numpy(T).to(jpos_pred.device)
            jpos_pred_world_rot = (Tz180[None, :3, :3] @ jpos_pred_world.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            jpos_pred_world_rot = jpos_pred_world
        j_pred_rot.append(jpos_pred_world_rot)
    
    j_pred_rot = torch.stack(j_pred_rot, dim=0)
    return j_pred_rot


def match_3d_joints(j_pred_rot, j_gts, debug=False):
    # make joints relative for matching in 3D

    if isinstance(j_pred_rot, torch.Tensor):
        j_pred_rot = j_pred_rot.cpu().numpy()
    if isinstance(j_gts, torch.Tensor):
        j_gts = j_gts.cpu().numpy()

    j_pred_rot_one = j_pred_rot[:, 0]
    j_pred_rel = j_pred_rot_one - j_pred_rot_one[:, 0:1]

    j_gts_one = j_gts[:, 0]
    j_gts_rel = j_gts_one - j_gts_one[:, 0:1]#.shape

    try:
        row_ind, col_ind = match_w_hungarian(j_pred_rel, j_gts_rel)
    except:
        return None, None

    return row_ind, col_ind


def get_dataset_data(ROOT, data_name, subj_name):

    sub_n = f"_{subj_name}" if subj_name != "." else ""
    chi3d_path = f"{ROOT}/data/{data_name}/{data_name}{sub_n}_embodied_cam2w_p1.pkl"
    print(f"\nReading GT PROC data from: {chi3d_path}")
    data_chi3d = read_pickle(chi3d_path)

    chi3d_path_2 = f"{ROOT}/data/{data_name}/{data_name}{sub_n}_embodied_cam2w_p2.pkl"
    data_chi3d_2 = read_pickle(chi3d_path_2)
    
    keys = sorted(data_chi3d.keys())
    print(f"\nKEYS from  GT PROC data are: {keys}\n")
    data_vars =  data_chi3d[keys[0]].keys()
    print(f"\nVARS from GT P1 PROC data are: {data_vars}")
    data_vars_2 =  data_chi3d_2[keys[0]].keys()
    print(f"VARS from GT P2 PROC data are: {data_vars_2}\n")

    return data_chi3d, data_chi3d_2





