import numpy as np
import torch
from uhc.smpllib.smpl_robot import Robot
from uhc.smpllib.torch_smpl_humanoid import Humanoid
import mujoco_py
from uhc.utils.config_utils.copycat_config import Config as CC_Config
import os.path as osp
from utils.pyquaternion import Quaternion as Q
from utils.misc import read_pickle
from utils.misc import write_pickle
from uhc.smpllib.smpl_mujoco import smpl_to_qpose_torch
from utils.smpl import smpl_to_verts
from utils.net_utils import get_hostname
from scipy.spatial.transform import Rotation as sRot
from tqdm import tqdm
from embodiedpose.models.humor.utils.humor_mujoco import MUJOCO_2_SMPL
from metrics.tools import match_w_hungarian
from metrics.prepare import parse_cam2world
from collections import defaultdict

hostname = get_hostname()


def get_pose_aa_from_slahmr_init_all(res_dict):
    # originally getting only the 1st frame info, but to override we need all
    B, T, _ = res_dict["pose_body"].shape
    body_pose_ = res_dict["pose_body"].cpu().numpy()
    betas = res_dict["betas"][:, None, :10].cpu().numpy()
    trans_orig = res_dict["trans"].cpu().numpy()
    global_orient = res_dict["root_orient"].cpu().numpy()
    body_pose = np.zeros([B, T, 69]).astype(np.float32)
    body_pose[:, :, :63] = body_pose_
    pose_aa = np.concatenate([global_orient, body_pose], axis=-1)
    return pose_aa, trans_orig, betas


def load_humanoid():
    print("Loading humanoid...")
    cc_cfg = CC_Config(cfg_id="copycat_eval_1", base_dir="./")
    smpl_robot = Robot(
        cc_cfg.robot_cfg,
        data_dir=osp.join(cc_cfg.base_dir, "data/smpl"),
        masterfoot=cc_cfg.masterfoot,
    )
    model = mujoco_py.load_model_from_xml(smpl_robot.export_xml_string().decode("utf-8"))
    humanoid = Humanoid(model=model)
    return smpl_robot, humanoid, cc_cfg


def get_camera_transform(seq, vid_name, return_gnd=False, data_name='chi3d'):
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
    final_Rt = Txm90[None] @ Rt_gnd_inv[None]  # @ T_c2w
    scene_dict['final_Rt'] = final_Rt
    scene_dict['res_dir'] = res_dir

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
    pelvis = joints[0, :, 0, None]  # (B, 1, 3)
    trans_i = trans_i[:, None]
    pelvis = pelvis.to(trans_i) - trans_i
    trans_i = (final_Rt[:, :3, :3] @ (trans_i + pelvis).permute(0, 2, 1)).permute(0, 2, 1) + final_Rt[:, None, :3,
                                                                                             3] - pelvis
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
    if body_dim == 69:
        pose_aa = torch.cat([root_aa, body_pose_aa], dim=-1)
    elif body_dim == 63:
        pose_aa = torch.cat([root_aa, body_pose_aa, torch.zeros(B, 6).to(root_aa)], dim=-1)
    else:
        raise ValueError(f"ERROR: Weird body_dim: {body_dim}!!")

    return_dict = {
        'pose_aa': pose_aa,
        'trans': trans,
        'shape': shape,
    }
    return return_dict


def jpos_pred_from_res_dict(res_dict):
    jpos_pred_ = res_dict["joints"]
    # SMPL-H joints, not SMPL
    jpos_pred = jpos_pred_[:, :, :24]  # these have one wrong joint, correct this!
    # this corrects the erroneous joint, due to the SMPL-H model
    jpos_pred[:, :, -1] = jpos_pred_[:, :, 37]
    return jpos_pred


def read_slahmr_transform_and_body(seq, vid_name, data_name='chi3d', sub_dir=None, sla_postfix=None):
    try:
        res_dict, Rx90_Rtgnd = get_camera_transform(seq, vid_name, return_gnd=True, data_name=data_name)
        # out = get_slahmr_emb2cam_transform(data_name, seq_name[4:], False, seq_num)
        # rotation, translation, focal_length, camera_center, _ = out
    except:
        print(f"skipping {vid_name}, transforms data does not exist! generate '*_scene_dict.pkl' in slahrm")
        return None, None
    # for now using only the first cam, but later use all
    final_Rt = res_dict['final_Rt']

    jpos_pred = jpos_pred_from_res_dict(res_dict)

    root_aa = res_dict['root_orient'].cpu()
    body_pose_aa = res_dict["pose_body"].cpu()
    B = root_aa.shape[0]
    seq_len = root_aa.shape[1]
    trans = res_dict["trans"].cpu()
    shape = res_dict["betas"][..., :10].cpu()

    # read post-optim results if desired
    if sub_dir is not None and sla_postfix is not None:
        res_dir = res_dict['res_dir']
        res_dir = res_dir.replace('-val', sla_postfix)
        try:
            new_res_dict = read_pickle(f"{res_dir}/{sub_dir}/{vid_name}_scene_dict.pkl")
        except:
            tqdm.write(f"skipping {sub_dir}/{vid_name}, _scene_dict does not exist!")
            return None, None
        try:
            # betas here are 0 as they are not optim, read from the "original" slahmr results
            pose_aa, trans_orig, _ = get_pose_aa_from_slahmr_init_all(new_res_dict)
            jpos_pred = jpos_pred_from_res_dict(new_res_dict)
            # ALSO betas have to be reversed in order, due to our embPose dataset having swap opt as defaut, todo change that
            shape = shape[[1, 0]]
        except:
            tqdm.write(f"skipping {vid_name}, does not exist! due to slahrm estimate (discard for now)")
            return None, None
        pose_aa = torch.tensor(pose_aa).float()
        # pose = pose_aa[0][0, 3:]
        trans = torch.tensor(trans_orig).float()
    else:
        pose_aa = torch.cat([root_aa, body_pose_aa, torch.zeros(B, seq_len, 6).to(root_aa)], dim=-1)

    return_dict = {
        'final_Rt': final_Rt,
        'jpos_pred': jpos_pred,
        'pose_aa': pose_aa,
        'trans': trans,
        'shape': shape,
        'Rx90_Rtgnd': Rx90_Rtgnd,
        # 'res_dict': res_dict,

    }
    return return_dict, seq_len


def parse_gt_joints_emb_format(emb_data, seq_len, jpos_pred):
    j_gts = []
    for p_id in range(len(jpos_pred[:2])):
        # NOTE: for some reason we have to rotate the joints and SMPL 180 over z-axis (check why).
        gt_jts = emb_data[p_id]['gt_jpos']
        seq_len_min = min(gt_jts.shape[0], seq_len)
        gt_jts = gt_jts[:seq_len_min].reshape([seq_len_min, -1, 3])
        gt_jts_smpl = gt_jts[:, MUJOCO_2_SMPL].copy()
        gt_jts = gt_jts_smpl.copy()
        j_gts.append(torch.tensor(gt_jts).float())
    j_gts = torch.stack(j_gts, dim=0)
    return j_gts, seq_len_min


def apply_transform(jpos_pred, final_Rt):
    jpos_pred_world = (final_Rt[:, :3, :3] @ jpos_pred.permute(0, 2, 1)).permute(0, 2, 1) + final_Rt[:, None, :3, 3]
    return jpos_pred_world


def rot_pose_180z(jpos_pred_world):
    """for torch"""
    T = Q(axis=[0, 0, 1], angle=np.pi).transformation_matrix.astype(np.float32)
    Tz180 = torch.from_numpy(T).to(jpos_pred_world.device)
    jpos_pred_world_rot = (Tz180[None, :3, :3] @ jpos_pred_world.permute(0, 2, 1)).permute(0, 2, 1)
    return jpos_pred_world_rot


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

    j_pred_rot_one = j_pred_rot[:, 0].cpu()
    j_pred_rel = j_pred_rot_one - j_pred_rot_one[:, 0:1]

    j_gts_one = j_gts[:, 0]
    j_gts_rel = j_gts_one - j_gts_one[:, 0:1]  # .shape

    j_pred_rel = j_pred_rel.numpy()
    j_gts_rel = j_gts_rel.numpy()

    try:
        row_ind, col_ind = match_w_hungarian(j_pred_rel, j_gts_rel)
    except:
        return None, None

    if debug:
        from utils.misc import save_pointcloud
        op = "inspect_out/prepare_slahmr/"
        idx = 5
        save_pointcloud(j_gts_rel[0], f"{op}/joints-rel_ord/{args.exp_name}/gt_jpos_0.ply")
        save_pointcloud(j_gts_rel[1], f"{op}/joints-rel_ord/{args.exp_name}/gt_jpos_1.ply")
        save_pointcloud(j_pred_rel[0], f"{op}/joints-rel_ord/{args.exp_name}/pred_jpos_0.ply")
        save_pointcloud(j_pred_rel[1], f"{op}/joints-rel_ord/{args.exp_name}/pred_jpos_1.ply")

    return row_ind, col_ind


def get_dataset_data(ROOT, data_name, subj_name):
    sub_n = f"_{subj_name}" if subj_name != "." else ""
    chi3d_path = f"{ROOT}/data/{data_name}/{data_name}{sub_n}_embodied_cam2w_p1.pkl"
    data_chi3d = read_pickle(chi3d_path)
    chi3d_path_2 = f"{ROOT}/data/{data_name}/{data_name}{sub_n}_embodied_cam2w_p2.pkl"
    data_chi3d_2 = read_pickle(chi3d_path_2)
    return data_chi3d, data_chi3d_2


T = Q(axis=[0, 0, 1], angle=np.pi).transformation_matrix.astype(np.float32)
Tz180 = torch.from_numpy(T).cuda()


def get_emb_paths(data_name, exp_name, slahmr_overwrite=False, get_add=False, sla_root=False):
    ROOT = "."
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


def get_smpl_gt_data(gt_chi3d):
    smpl_dict_gt, smpl_gt_jts = [], []
    for n, chi3d_data in enumerate(gt_chi3d):
        this_dict = {
            'pose_aa': chi3d_data['pose_aa'].float(),
            'trans': chi3d_data['trans'].float(),
            'betas': chi3d_data['shape'][0, None].float(),
        }
        (_, joints), _ = smpl_to_verts(this_dict['pose_aa'], this_dict['trans'],
                                       betas=this_dict['betas'], return_joints=True)
        gt_jts_smpl_this = joints[0, :, :24].numpy().reshape(-1, 72)
        smpl_gt_jts.append(gt_jts_smpl_this)
        smpl_dict_gt.append(this_dict)
    smpl_gt_jts = np.stack(smpl_gt_jts, axis=0).reshape([2, -1, 24, 3])
    smpl_gt_jts = torch.tensor(smpl_gt_jts).float()
    return smpl_dict_gt, smpl_gt_jts


def parse_emb_gt_data(data_chi3d_p1, data_chi3d_p2, sname):
    gt_data_p1 = data_chi3d_p1[sname]
    gt_data_p2 = data_chi3d_p2[sname]
    gt_p1 = parse_smpl_data(gt_data_p1)
    gt_p2 = parse_smpl_data(gt_data_p2)
    gt_chi3d = [gt_p1, gt_p2]
    Rt = parse_cam2world(gt_data_p1)
    return gt_chi3d, Rt


def get_emb_robot_models(mean_shape, smpl_robot):
    models = []
    smpl_robot.load_from_skeleton(mean_shape[0, None], gender=[0], objs_info=None)
    model = mujoco_py.load_model_from_xml(smpl_robot.export_xml_string().decode("utf-8"))
    models.append(model)

    smpl_robot.load_from_skeleton(mean_shape[1, None], gender=[0], objs_info=None)
    model = mujoco_py.load_model_from_xml(smpl_robot.export_xml_string().decode("utf-8"))
    models.append(model)
    return models


def transform_coords_smpl_data(p_id, pose_aa, trans, mean_shape, final_Rt, device):
    # pose_aa is (B, 72), trans is (B, 3), needs to be converted to world
    # transform to world coords with camera and ground plane orientation, final_Rt contains both
    pose_aa_i = pose_aa[p_id]
    trans_i = trans[p_id].clone().to(device)
    betas = mean_shape[None, p_id, :10].to(device)
    pose_aa_w_i = rot_pose(pose_aa_i.cpu(), final_Rt.cpu()).to(device)
    trans_w_i, verts_i_w, faces = correct_smplx_offset(pose_aa_w_i, trans_i, betas, final_Rt, get_verts=False)
    # qpos = smpl_to_qpose_torch(pose_aa_w_i, model, trans=trans_w_i, count_offset=True)
    pose_aa_w_rot_i = rot_pose(pose_aa_w_i.cpu(), Tz180[None].cpu()).to(trans_i)
    trans_w_rot_i, verts_w_rot, faces = correct_smplx_offset(pose_aa_w_rot_i, trans_w_i, betas, Tz180[None],
                                                             get_verts=True)

    return pose_aa_w_rot_i, trans_w_rot_i, betas


def prepare_slahmr(args):
    """
    code intended to
    prepare_slahmr_results for computing metrics and comp to baseline,
    here I add the GT to the results dict from slahmr.
    The GT comes from embPose results, meaning that they are taken from the simulation's agent.
    The GT is read from results folders from EmbPose.
    They are generated in eval_scene_multi using:
                                gt_jpos = self.env.gt_targets[n]['wbpos'][self.env.cur_t].copy() # shape: (72,)
                            res[n]["gt_jpos"].append(gt_jpos)
    from qpos you can get wbpos or the same: jpos
        gt_targets = self.humanoid[n].qpos_fk(torch.from_numpy(gt_qpos))
        self.gt_targets = self.smpl_humanoid.qpos_fk_batch(self.ar_context["qpos"])
    """

    debug = args.debug
    data_name = args.data_name
    smpl_robot, humanoid, cc_cfg = load_humanoid()
    path, SEQS, ROOT, RES_ROOT = get_emb_paths(data_name, args.exp_name)

    print(f"SUB_DIR: {args.sub_dir}")
    print(f"SLA_POSTFIX: {args.sla_postfix}")

    for subj_name in SEQS:
        print(f"Parsing data {subj_name}...")
        # get the GT from dataset processed files
        sname = subj_name if data_name == 'chi3d' or data_name == 'expi' else '.'
        data_chi3d_p1, data_chi3d_p2 = get_dataset_data(ROOT, data_name, sname)

        results_dict = defaultdict(list)
        results_dict_verts = defaultdict(list)
        # for seq_name in chi3d_names:
        for n, seq_name in enumerate(tqdm(data_chi3d_p1)):

            if args.filter_seq is not None:
                if seq_name != args.filter_seq:
                    print("*** WARNING: filtering seqs")
                    continue
            if debug and n > 2:
                break
            # seq_name = 's02_Grab_17'
            # results_dict[seq_name] = []
            tqdm.write(f"{args.exp_name} - {subj_name} - {seq_name}")
            vid_name = "_".join(seq_name.split('_')[1:3]) if data_name == 'chi3d' or data_name == 'expi' else seq_name
            device = torch.device("cuda:0")

            # read SLAHMR data: pose, joints and transform
            return_dict, seq_len = read_slahmr_transform_and_body(sname, vid_name, data_name,
                                                                  sub_dir=args.sub_dir, sla_postfix=args.sla_postfix)

            if return_dict is None:
                print("failed to read slhamr results, skipping")
                continue
            final_Rt = return_dict["final_Rt"]  # (1, 4, 4)
            jpos_pred = return_dict["jpos_pred"]  # this comes from res_dict["joints"] from slahmr
            pose_aa = return_dict["pose_aa"]
            trans = return_dict["trans"]
            shape = return_dict["shape"]
            # res_dict = return_dict["res_dict"]

            mean_shape = shape[:, 0:10]

            try:
                models = get_emb_robot_models(mean_shape, smpl_robot)
            except:
                print("ERROR: probably got estimate for only one person!")

            seq_n = seq_name[:-6] if data_name == 'expi' else seq_name
            # parse GT data
            try:
                gt_chi3d, Rt = parse_emb_gt_data(data_chi3d_p1, data_chi3d_p2, seq_n)
            except:
                seq_n = seq_name
                gt_chi3d, Rt = parse_emb_gt_data(data_chi3d_p1, data_chi3d_p2, seq_n)

            smpl_dict_gt, smpl_gt_jts = get_smpl_gt_data(gt_chi3d)

            if args.data_name == 'hi4d':
                # for hi4d the GT cam is world2cam, need to invert. for chi3d, hoewever, it is already cam2world
                cam2world = torch.linalg.inv(Rt)
            else:
                cam2world = Rt

            # if I use directly the SMPL jts then I don't need to rotate them
            j_pred_world = transform_pred_joints(jpos_pred, cam2world, rot_180z=False)

            if args.data_name == 'hi4d':
                Tx90 = Q(axis=[1, 0, 0], angle=np.pi / 2).transformation_matrix.astype(np.float32)
                Tz180 = Q(axis=[0, 0, 1], angle=np.pi).transformation_matrix.astype(np.float32)
                Tr = Tz180[None] @ Tx90[None]
                Tr = torch.from_numpy(Tr).float().cuda()
                j_pred_world = apply_transform(j_pred_world.reshape(-1, 24, 3), Tr).reshape(2, -1, 24, 3)

            # get j_gts from embdata
            # j_gts, seq_len_min = parse_gt_joints_emb_format(emb_data, seq_len, jpos_pred)

            if debug:
                from utils.joints3d import joints_to_skel
                op = f"inspect_out/prepare_slahmr/{args.data_name}/{seq_name}"
                idx = 100
                joints_to_skel(jpos_pred[:, idx].cpu(), f"{op}/skels_pred_{idx:03d}.ply", format='smpl', radius=0.015)
                joints_to_skel(j_pred_world[:, idx].cpu(), f"{op}/skels_world_{idx:03d}.ply", format='smpl',
                               radius=0.015)
                joints_to_skel(smpl_gt_jts[:, idx].cpu(), f"{op}/skels_smplgt_{idx:03d}.ply", format='smpl',
                               radius=0.015)

            # match pred poses with GT poses, they are not in the same order
            # match j_gts from embdata  w/ slahmr preds in relative 3D space
            # row_ind, col_ind = match_3d_joints(j_pred_rot, j_gts)
            # match smpl to slahmr
            row_ind_smpl, col_ind_smpl = match_3d_joints(j_pred_world, smpl_gt_jts)
            seq_len_min = min(smpl_gt_jts.shape[1], seq_len)

            if row_ind_smpl is None:
                print(f"matching failed for {seq_name}! skipping...")
                continue

            # do matching to get each correct kpts, copy slahmr code
            for p_id in range(len(jpos_pred[:2])):
                # NOTE: for some reason we have to rotate the joints and SMPL 180 over z-axis --> only when using gt from embpose.
                # these are now directly from smpl
                gt_jts = smpl_gt_jts[col_ind_smpl[p_id]]

                model = models[p_id]
                pose_aa_w_rot_i, trans_w_rot_i, betas = transform_coords_smpl_data(p_id, pose_aa, trans, mean_shape,
                                                                                   final_Rt, device)
                qpos_rot = smpl_to_qpose_torch(pose_aa_w_rot_i.cuda(), model, trans=trans_w_rot_i, count_offset=False)
                qpos_rot = qpos_rot.cpu().numpy()

                gt_data = {}
                gt_data_verts = {}

                # qpos use only to compute the root metric
                # gt_data['gt'] = emb_data[col_ind_smpl[p_id]]['gt'][:seq_len_min] #.cpu().numpy()
                # BEWARE: dummy value as this metric is not important for now
                gt_data['gt'] = qpos_rot[:seq_len_min]
                gt_data['pred'] = qpos_rot[:seq_len_min]
                # joints these are used to compute MPJPE
                gt_jts_ = gt_jts[:seq_len_min].cpu().numpy()
                gt_data['gt_jpos'] = gt_jts_.reshape(-1, 72)
                # gt_data['pred_jpos'] = jpos_pred_world_rot[:seq_len_min].cpu().numpy()
                j_pred_r = j_pred_world[p_id, :seq_len_min].cpu().numpy()
                gt_data['pred_jpos'] = j_pred_r.reshape(-1, 72)
                # other
                gt_data['percent'] = 1.0
                gt_data['fail_safe'] = False

                # convert slahrm prediction to wbpos/jpos
                humanoid_fk = Humanoid(model=model)
                qpos_rot_t = torch.from_numpy(qpos_rot).float()
                fk_result = humanoid_fk.qpos_fk(qpos_rot_t, to_numpy=False)
                pred_wbpos = fk_result['wbpos'].numpy()
                # pred_wbpos = pred_wbpos.reshape(-1, 24, 3)
                gt_data['pred_jpos_wbpos'] = pred_wbpos

                smpl_dict = {
                    'pose_aa': pose_aa_w_rot_i.float(),
                    'trans': trans_w_rot_i.float(),
                    'betas': betas.float(),
                }

                smpl_dict_gt = {
                    'pose_aa': gt_chi3d[col_ind_smpl[p_id]]['pose_aa'].float(),
                    'trans': gt_chi3d[col_ind_smpl[p_id]]['trans'].float(),
                    'betas': gt_chi3d[col_ind_smpl[p_id]]['shape'][0, None].float(),
                }
                gt_data['smpl_pred'] = smpl_dict
                gt_data['smpl_gt'] = smpl_dict_gt
                gt_data['gt_jpos_smpl'] = smpl_gt_jts[col_ind_smpl[p_id]][:seq_len_min]
                # save file
                results_dict[seq_name].append(gt_data)

                if args.save_verts:
                    verts, faces = smpl_to_verts(pose_aa_w_rot_i, trans_w_rot_i, betas=mean_shape[p_id, None, :10])
                    gt_data_verts['pred_vertices'] = verts[0].cpu().numpy()
                    # save file
                    results_dict_verts[seq_name].append(gt_data_verts)

        # add subdir for diff slahmr post optims
        # path = f"{path}/{args.sub_dir}" if args.sub_dir is not None else path
        # save the results
        debug_n = '' if not debug else f'_debug'
        outpath = path + f"/{subj_name}_slahmr{debug_n}.pkl"
        write_pickle(results_dict, outpath)
        print(f"file saved to {outpath}")

        if args.save_verts:
            outpath_verts = str(outpath).replace(".pkl", "_verts.pkl")
            write_pickle(results_dict_verts, outpath_verts)
            print(f"file saved to {outpath}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', type=int, choices=[0, 1], default=0)
    parser.add_argument('--subj_name', type=str, default='.')
    parser.add_argument('--data_name', type=str, choices=['chi3d', 'hi4d', 'expi'], default='chi3d')
    parser.add_argument("--exp_name", type=str, default='normal_op')  # this has to be fixed here
    parser.add_argument('--filter_seq', type=str, default=None)
    parser.add_argument("--save_verts", type=int, choices=[0, 1], default=0)
    parser.add_argument("--sub_dir", type=str, default=None)
    parser.add_argument("--sla_postfix", type=str, default=None)

    args = parser.parse_args()

    prepare_slahmr(args)
