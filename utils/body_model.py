# Copyright (C) 2022-2023 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import torch
import torch.nn as nn
import roma
from utils.threed.skeleton import get_smpl_pose, get_smplh_pose, get_smplx_pose

def pose_to_vertices(y, pose_type, alpha, bm, parallel=False, manual_seq_len=None, resize=True, return_joints=False,
                     body_pose=False, betas=None):
    """
    Map SMPL-H/X parameters to a human mesh
    Args:
        - y: [batch_size, seq_len, K]
    Return:
        - verts: [batch_size, seq_len, n_vertices, 3]
    """
    batch_size, seq_len, *_ = y.size()

    if not parallel:
        list_verts = []
        list_jts = []
        list_bpose = []
        for i in range(batch_size):
            pose = eval(f"get_{pose_type}_pose")(y[i])
            if betas is not None:
                assert isinstance(betas, torch.Tensor), 'betas must be a tensor no numpy!'
                if batch_size > 1:
                    betas_ = betas[i]
                    assert len(betas_.shape)==2 and betas_.shape[0]==1, 'betas must be a tensor of SHAPE [1, 10] per batch! so MAKE it [B, 1, 10]'
                    pose.update({'betas': betas_})
                else:
                    pose.update({'betas': betas})
            if alpha == 0:
                if 'trans' in pose:
                    pose['trans'] *= 0.
                elif 'transl' in pose:
                    pose['transl'] *= 0.
            # for k,v in pose.items():
                # print(k+':'+str(v.shape))
            if manual_seq_len is not None:
                pose.update({'betas': torch.zeros((manual_seq_len, 10)).cuda(),
                             'batch_size': manual_seq_len})
            if return_joints:
                output = bm(**pose, return_full_pose=True)
                verts = output.vertices
                jts = output.joints
                list_jts.append(jts)
                bo_pose = output.body_pose
                list_bpose.append(bo_pose)
            else:
                verts = bm(**pose).vertices
            list_verts.append(verts)

        verts = torch.stack(list_verts)
        if return_joints:
            verts = torch.stack(list_verts)
            if body_pose:
                return verts, torch.stack(list_jts), torch.stack(list_bpose)
            return verts, torch.stack(list_jts)
        else:
            return verts
    else:
        cy = y.reshape((-1, y.shape[-1])) # y is pose input [B, T, 168]
        pose = eval(f"get_{pose_type}_pose")(cy) # cy is [B*T, 168
        if alpha == 0:
            if 'trans' in pose:
                pose['trans'] *= 0.
            elif 'transl' in pose:
                pose['transl'] *= 0.
        # for k,v in pose.items():
            # print(k+':'+str(v.shape))
        if manual_seq_len is not None:
            pose.update({'betas': torch.zeros((manual_seq_len,10)).cuda(),
                             'batch_size': manual_seq_len})
        if return_joints:
            output = bm(**pose, return_full_pose=True)
            verts = output.vertices
            jts = output.joints
            bo_pose = output.body_pose
        else:
            out = bm(**pose, return_full_pose=True)
            verts = out.vertices
            bo_pose = out.body_pose
        
        if resize:
            verts = verts.reshape((y.shape[0], y.shape[1], verts.shape[-2], verts.shape[-1]))
            if return_joints:
                jts = jts.reshape((y.shape[0], y.shape[1], jts.shape[-2], jts.shape[-1]))
                if body_pose:
                    return verts, jts, bo_pose
                return verts, jts
        if body_pose and not return_joints:
            return verts, bo_pose

        return verts


def get_trans(delta, valid):
    """ Compute absolute translation coordinates from deltas """
    # t=0 is left unchanged
    trans = [delta[:, 0].clone()]
    for i in range(1, delta.size(1)):
        if valid is not None:
            assert valid.shape[1] > i, "Here is a bug"
            d = delta[:, i] * valid[:, [i]].float()
        else:
            d = delta[:, i]
        trans.append(trans[-1] + d)
    trans = torch.stack(trans, 1)
    return trans


def six_dim(x):
    """ Move to 6d representation and represent translations as deltas """
    batch_size, seq_len, _ = x.size()

    # move to 6d representation
    x = x.reshape(batch_size, seq_len, -1, 3)
    trans = x[:, :, -1]  # [batch_size,seq_len,3]
    rotvec = x[:, :, :-1]
    rotmat = roma.rotvec_to_rotmat(rotvec)  # [batch_size,seq_len,n_jts,3,3]
    x = torch.cat([rotmat[..., :2].flatten(2), trans], -1)  # [batch_size,seq_len,n_jts*6+3]
    return x, rotvec, rotmat, trans


class SimplePreparator(nn.Module):
    def __init__(self, mask_trans, pred_trans, **kwargs):
        super(SimplePreparator, self).__init__()
        self.mask_trans = mask_trans
        self.pred_trans = pred_trans

    def forward(self, x, **kwargs):
        x, rotvec, rotmat, trans = six_dim(x)
        assert self.mask_trans == (not self.pred_trans), "mask_trans and pred_trans incoherent"
        if self.mask_trans:
            trans = torch.zeros_like(trans)
        trans_delta = trans[:, 1:] - trans[:, :-1]
        return x, rotvec, rotmat, trans, trans_delta, None
