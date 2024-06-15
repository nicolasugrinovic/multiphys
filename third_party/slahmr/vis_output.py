import numpy as np
import torch

# from body_model import run_smpl
from slahmr.geometry import camera as cam_util
from geometry.plane import parse_floor_plane, get_plane_transform

from util.tensor import detach_all, to_torch, move_to

# from .tools import smpl_to_geometry


def prep_result_vis(world_smpl, res, vis_mask,
                    # track_ids,
                    # body_model

                    ):
    """
    :param res (dict) with (B, T, *) tensor elements, B tracks and T frames
    :param vis_mask (B, T) with visibility of each track in each frame
    :param track_ids (B,) index of each track
    """
    print("RESULT FIELDS", res.keys())
    res = detach_all(res)
    # with torch.no_grad():
    #     world_smpl = run_smpl(
    #         body_model,
    #         res["trans"],
    #         res["root_orient"],
    #         res["pose_body"],
    #         res.get("betas", None),
    #     )
    T_w2c = None
    floor_plane = None
    if "cam_R" in res and "cam_t" in res:
        T_w2c = cam_util.make_4x4_pose(res["cam_R"][0], res["cam_t"][0])
    if "floor_plane" in res:
        floor_plane = res["floor_plane"][0]

    # NOTE: inside build_scene_dict there is all I need to transform the poses into the world coordinate system
    return build_scene_dict(
        world_smpl,
        vis_mask,
        # track_ids,
        T_w2c=T_w2c,
        floor_plane=floor_plane,
    )


def build_scene_dict(
    world_smpl,
        vis_mask,
        # track_ids,
        T_w2c=None, floor_plane=None, **kwargs
):
    scene_dict = {}

    # first get the geometry of the people
    # lists of length T with (B, V, 3), (F, 3), (B, 3)
    # scene_dict["geometry"] = smpl_to_geometry(
    #     world_smpl["vertices"], world_smpl["faces"], vis_mask, track_ids
    # )

    if T_w2c is None:
        T_w2c = torch.eye(4)[None]

    T_c2w = torch.linalg.inv(T_w2c) # T_w2c is (B, 4, 4) --> R|t in homogeneous coordinates
    # rotate the camera slightly down and translate back and up
    T = cam_util.make_4x4_pose(
        cam_util.rotx(-np.pi / 10), torch.tensor([0, -1, -2])
    ).to(T_c2w.device)

    scene_dict["cameras"] = {
        "src_cam": T_c2w,
        "front": torch.einsum("ij,...jk->...ik", T, T_c2w),
    }
    # tranf_c = torch.einsum("ij,...jk->...ik", T, T_c2w) , tranf_c[0], T_c2w[0]
    if floor_plane is not None:
        # compute the ground transform
        # use the first appearance of a track as the reference point
        # tid, sid = torch.where(vis_mask > 0)
        # idx = tid[torch.argmin(sid)]
        idx = 0
        # use the smpl root
        root = world_smpl["joints"][idx, 0, 0].detach().cpu()
        floor = parse_floor_plane(floor_plane.detach().cpu())
        R, t = get_plane_transform(torch.tensor([0.0, 1.0, 0.0]), floor, root)
        scene_dict["ground"] = cam_util.make_4x4_pose(R, t)

    return scene_dict

