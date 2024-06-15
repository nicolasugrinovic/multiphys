import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from body_model import SMPL_JOINTS, KEYPT_VERTS, smpl_to_openpose, run_smpl
from geometry.rotation import rotation_matrix_to_angle_axis
from embpose_util.logger import Logger
from embpose_util.tensor import move_to, detach_all

from .helpers import estimate_initial_trans
from .params import CameraParams
from slahmr.geometry import camera as cam_util
from utils.smpl import smpl_to_verts
from uhc.smpllib.np_smpl_humanoid_batch import smpl_op_to_op
from embpose_util.tensor import move_to, detach_all, to_np, to_torch
from utils.smpl import smpl_to_verts_bs
from optim_steps.optim_tools import perspective_projection_w_cam

J_BODY = len(SMPL_JOINTS) - 1  # no root

smpl2op_map = smpl_to_openpose(
    "smpl",
    use_hands=False,
    use_face=False,
    use_face_contour=False,
    openpose_format="coco25",
)


class BaseSceneModel(nn.Module):
    """
    Scene model of sequences of human poses.
    All poses are in their own INDEPENDENT camera reference frames.
    A basic class mostly for testing purposes.

    Parameters:
        batch_size:  number of sequences to optimize
        seq_len:     length of the sequences
        body_model:  SMPL body model
        pose_prior:  VPoser model
        fit_gender:  gender of model (optional)
    """

    def __init__(
        self,
        batch_size,
        seq_len,
        body_model,
        pose_prior,
        fit_gender="neutral",
        use_init=False,
        opt_cams=False,
        opt_scale=True,
        **kwargs,
    ):
        super().__init__()
        B, T = batch_size, seq_len
        self.batch_size = batch_size
        self.seq_len = seq_len

        self.body_model = body_model
        self.fit_gender = fit_gender

        self.pose_prior = pose_prior
        self.latent_pose_dim = self.pose_prior.latentD

        self.num_betas = body_model.bm.num_betas

        self.smpl2op_map = smpl_to_openpose(
            self.body_model.model_type,
            use_hands=False,
            use_face=False,
            use_face_contour=False,
            openpose_format="coco25",
        )

        self.use_init = use_init
        print("USE INIT", use_init)
        self.opt_scale = opt_scale
        self.opt_cams = opt_cams
        print("OPT SCALE", self.opt_scale)
        print("OPT CAMERAS", self.opt_cams)
        self.params = CameraParams(batch_size)

    def initialize(self, obs_data, cam_data, cam):
        """
        cam is the embpose cam, modified NUK
        """
        Logger.log("Initializing scene model with observed data")

        # initialize cameras
        self.params.set_cameras(
            cam_data,
            opt_scale=self.opt_scale,
            opt_cams=self.opt_cams,
            opt_focal=self.opt_cams,
        )

        # initialize body params
        B, T = self.batch_size, self.seq_len
        init_pose = torch.zeros(B, T, self.latent_pose_dim)
        init_betas = torch.zeros(B, self.num_betas)
        init_trans = torch.zeros(B, T, 3)
        init_rot = (
            torch.tensor([np.pi, 0, 0], dtype=torch.float32)
            .reshape(1, 1, 3)
            .repeat(B, T, 1)
        )

        if self.use_init and "init_body_pose" in obs_data:
            init_pose_aa = obs_data["init_body_pose"][:, :, :J_BODY, :]
            init_pose = self.pose2latent(init_pose_aa)

        if self.use_init and "init_root_orient" in obs_data:
            init_rot = obs_data["init_root_orient"]

        if self.use_init and "init_betas" in obs_data:
            init_betas = obs_data["init_betas"]

        if self.use_init and "init_trans" in obs_data:
            init_trans = obs_data["init_trans"]  # (B, T, 3)
            # transform into world frame (T, 3, 3), (T, 3)
            # do not do this because of EmbPose frame
            # R_w2c, t_w2c = cam_data["cam_R"], cam_data["cam_t"]
            # R_c2w = R_w2c.transpose(-1, -2)
            # t_c2w = -torch.einsum("tij,tj->ti", R_c2w, t_w2c)
            # init_trans = torch.einsum("tij,btj->bti", R_c2w, init_trans) + t_c2w[None]
        else:
            # initialize trans with reprojected joints
            # body_pose = self.latent2pose(init_pose)
            # pred_data = self.pred_smpl(init_trans, init_rot, body_pose, init_betas)
            # nice to have func. analice later
            # init_trans = estimate_initial_trans(
            #     body_pose,
            #     pred_data["joints3d_op"],
            #     obs_data["joints2d"],
            #     obs_data["intrins"][:, 0],
            # )
            print('inside else in BaseModel init')

        self.params.set_param("latent_pose", init_pose)
        self.params.set_param("betas", init_betas)
        self.params.set_param("trans", init_trans)
        self.params.set_param("root_orient", init_rot)
        self.embpose_cam = cam

        self.init_data = {
            "init_pose_aa": init_pose_aa.clone(),
            "init_rot": init_rot.clone(),
            "init_betas": init_betas.clone(),
            "init_trans": init_trans.clone(),
            "latent_pose": init_pose.clone(),
        }


    def get_optim_result(self, **kwargs):
        """
        Collect predicted outputs (latent_pose, trans, root_orient, betas, body pose) into dict
        """
        res = self.params.get_dict()
        if "latent_pose" in res:
            res["pose_body"] = self.latent2pose(self.params.latent_pose).detach()

        # add the cameras
        res["cam_R"], res["cam_t"], _, _ = self.params.get_cameras()
        res["intrins"] = self.params.intrins
        return {"world": res}

    def latent2pose(self, latent_pose):
        """
        Converts VPoser latent embedding to aa body pose.
        latent_pose : B x T x D
        body_pose : B x T x J*3
        """
        B, T, _ = latent_pose.size()
        d_latent = self.pose_prior.latentD
        latent_pose = latent_pose.reshape((-1, d_latent))
        body_pose = self.pose_prior.decode(latent_pose, output_type="matrot")
        body_pose = rotation_matrix_to_angle_axis(
            body_pose.reshape((B * T * J_BODY, 3, 3))
        ).reshape((B, T, J_BODY * 3))
        return body_pose

    def pose2latent(self, body_pose):
        """
        Encodes aa body pose to VPoser latent space.
        body_pose : B x T x J*3
        latent_pose : B x T x D
        """
        B, T = body_pose.shape[:2]
        body_pose = body_pose.reshape((-1, J_BODY * 3))
        latent_pose_distrib = self.pose_prior.encode(body_pose)
        d_latent = self.pose_prior.latentD
        latent_pose = latent_pose_distrib.mean.reshape((B, T, d_latent))
        return latent_pose

    def pred_smpl(self, trans, root_orient, body_pose, betas):
        """
        Forward pass of the SMPL model and populates pred_data accordingly with
        joints3d, verts3d, points3d.

        trans : B x T x 3
        root_orient : B x T x 3
        body_pose : B x T x J*3
        betas : B x D
        """
        smpl_out = run_smpl(self.body_model, trans, root_orient, body_pose, betas)
        joints3d, points3d = smpl_out["joints"], smpl_out["vertices"]

        # select desired joints and vertices
        joints3d_body = joints3d[:, :, : len(SMPL_JOINTS), :]
        joints3d_op = joints3d[:, :, self.smpl2op_map, :]
        # hacky way to get hip joints that align with ViTPose keypoints
        # this could be moved elsewhere in the future (and done properly)
        joints3d_op[:, :, [9, 12]] = (
            joints3d_op[:, :, [9, 12]]
            + 0.25 * (joints3d_op[:, :, [9, 12]] - joints3d_op[:, :, [12, 9]])
            + 0.5
            * (
                joints3d_op[:, :, [8]]
                - 0.5 * (joints3d_op[:, :, [9, 12]] + joints3d_op[:, :, [12, 9]])
            )
        )
        verts3d = points3d[:, :, KEYPT_VERTS, :]

        return {
            "points3d": points3d,  # all vertices
            "verts3d": verts3d,  # keypoint vertices
            "joints3d": joints3d_body,  # smpl joints
            "joints3d_op": joints3d_op,  # OP joints
            "faces": smpl_out["faces"],  # index array of faces
        }

    def pred_params_smpl(self, pred_data=None):
        body_pose = self.latent2pose(self.params.latent_pose)
        pred_data = self.pred_smpl(
            self.params.trans, self.params.root_orient, body_pose, self.params.betas
        )
        return pred_data

    def get_curr_pose(self, pred_data=None):
        if pred_data is None:
            pred_data = self.params.get_vars()
        if 'world' in pred_data:
            pred_data = pred_data['world']
            body_pose = pred_data['pose_body']
        else:
            body_pose = self.latent2pose(pred_data['latent_pose'])
        root_orient = pred_data['root_orient'].squeeze(2)
        B, T, *_ = body_pose.shape
        trans = pred_data['trans']
        betas = pred_data['betas']
        pose_aa = torch.cat([root_orient, body_pose, torch.zeros(B, T, 6).to(root_orient)], dim=-1)

        pose_data = {
            'pose_aa': pose_aa,
            'trans': trans,
            'betas': betas,
        }

        return pose_data

    def get_init_pose_v1(self):
        init_pose_aa = self.init_data['init_pose_aa']
        init_rot = self.init_data['init_rot']
        init_trans = self.init_data['init_trans']
        init_betas = self.init_data['init_betas']
        latent_pose = self.init_data['latent_pose']

        B, T, *_ = init_pose_aa.shape

        pose_aa = torch.cat(
            [init_rot.reshape(B, T, -1), init_pose_aa.reshape(B, T, -1), torch.zeros(B, T, 6).to(init_rot)], dim=-1
        )

        with torch.no_grad():
            # body_pose = self.latent2pose(latent_pose)
            body_pose = init_pose_aa.reshape(B, T, -1)
            pred_data = self.pred_smpl(
                init_trans, init_rot, body_pose, init_betas
            )

            (_, joints1), _ = smpl_to_verts(pose_aa[0], init_trans[0], init_betas[0, None], return_joints=True)
            (_, joints2), _ = smpl_to_verts(pose_aa[1], init_trans[1], init_betas[1, None], return_joints=True)

        joints1_ = joints_to_emb12(joints1)
        joints2_ = joints_to_emb12(joints2)
        joints3d_emb = np.concatenate([joints1_, joints2_], 0)
        joints3d_emb = move_to(to_torch(joints3d_emb), 'cuda')

        pose_data = {
            'pose_aa': pose_aa,
            'trans': init_trans,
            'betas': init_betas,
        }

        pred_data["cameras"] = self.params.get_cameras()
        joints2d = cam_util.reproject(
            pred_data["joints3d_op"], *pred_data["cameras"]
        )
        pred_data["joints2d"] = joints2d
        #  cameras is cam_R, cam_t, cam_f, cam_center
        self.embpose_cam = move_to(to_torch(self.embpose_cam), 'cuda')
        R = self.embpose_cam['full_R']
        tr = self.embpose_cam['full_t']
        K = self.embpose_cam['K']
        R_ = R.expand(B, 1, 3, 3)
        tr_ = tr.expand(B, 1, 3)
        K_ = K.expand(1, 3, 3)
        f = torch.stack([K_[:, 0, 0], K_[:, 1, 1]], 1)
        center = torch.stack([K_[:, 0, 2], K_[:, 1, 2]], 1)
        cam = [R_, tr_, f, center]
        joints2d_emb_cam = cam_util.reproject(
            joints3d_emb, *cam
        )
        pred_data["joints2d_emb_cam"] = joints2d_emb_cam

        joints2d_emb = cam_util.reproject(
            joints3d_emb, *pred_data["cameras"]
        )
        pred_data["joints2d_emb"] = joints2d_emb

        pose_data.update(pred_data)

        (verts, joints_smpl), faces = smpl_to_verts_bs(pose_aa, init_trans, init_betas[:, None],
                                                       batch_size=2, return_joints=True)
        joints_smpl_op = joints_smpl[:, :, smpl2op_map, :]
        self.embpose_cam = to_np(move_to(self.embpose_cam, 'cpu'))
        j2d_init_emb1 = perspective_projection_w_cam(joints_smpl_op[0], self.embpose_cam, device='cpu')
        j2d_init_emb2 = perspective_projection_w_cam(joints_smpl_op[1], self.embpose_cam, device='cpu')
        j2d_init_emb = torch.stack([j2d_init_emb1, j2d_init_emb2])
        # j2d_init_emb = j2d_init_emb.permute(1, 0, 2, 3)

        pose_data["joints2d_emb_smpl_model"] = j2d_init_emb

        return pose_data

    def get_init_pose(self):
        init_pose_aa = self.init_data['init_pose_aa']
        init_rot = self.init_data['init_rot']
        init_trans = self.init_data['init_trans']
        init_betas = self.init_data['init_betas']
        latent_pose = self.init_data['latent_pose']

        B, T, *_ = init_pose_aa.shape

        pose_aa = torch.cat(
            [init_rot.reshape(B, T, -1), init_pose_aa.reshape(B, T, -1), torch.zeros(B, T, 6).to(init_rot)], dim=-1
        )

        with torch.no_grad():
            # body_pose = self.latent2pose(latent_pose)
            body_pose = init_pose_aa.reshape(B, T, -1)
            pred_data = self.pred_smpl(
                init_trans, init_rot, body_pose, init_betas
            )

            (verts, joints_smpl), faces = smpl_to_verts_bs(pose_aa, init_trans, init_betas[:, None],
                                                           batch_size=2, return_joints=True)
        joints3d_emb = joints_to_emb12(joints_smpl)
        joints3d_emb = move_to(to_torch(joints3d_emb), 'cuda')

        pose_data = {
            'pose_aa': pose_aa,
            'trans': init_trans,
            'betas': init_betas,
        }

        pred_data["cameras"] = self.params.get_cameras()
        joints2d = cam_util.reproject(
            pred_data["joints3d_op"], *pred_data["cameras"]
        )
        joints2d_emb = cam_util.reproject(
            joints3d_emb, *pred_data["cameras"]
        )

        pred_data["joints2d"] = joints2d
        pred_data["joints2d_emb"] = joints2d_emb
        pose_data.update(pred_data)

        return pose_data


def joints_to_emb12(joints1):
    smpl2op_map = np.array(
        [52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62])
    openpose_subindex = smpl2op_map < 22
    smpl2op_partial = smpl2op_map[openpose_subindex]
    joints1_ = joints1[:, :, smpl2op_partial]
    joints1_ = smpl_op_to_op(joints1_)
    return joints1_