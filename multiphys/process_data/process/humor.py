from pathlib import Path
from utils.misc import read_pickle
from utils.misc import write_pickle
import torch
from uhc.utils.transform_utils import (convert_aa_to_orth6d, rotation_matrix_to_angle_axis)
import numpy as np


path = "/home/nugrinovic/code/CVPR_2024/EmbodiedPose/sample_data/humor/for_embodied_demo/demo_clip.pkl"
data = read_pickle(path)
data.keys()
# convert to pose_6d
pose = data['pose']
pose_aa = np.zeros([pose.shape[0], 72])
pose_aa[:, :66] = pose

pose_aa = torch.from_numpy(pose_aa).float()
pose6d = convert_aa_to_orth6d(pose_aa)
pose_seq_6d = pose6d.reshape(pose6d.shape[0], 144)

output = {
    "pose_aa": pose_aa,
    "pose_6d": pose_seq_6d,
    "trans": data["trans"],
    "betas": data["shape"],
    "joints2d": data["joints2d"],
    }

write_pickle(output, "/home/nugrinovic/code/CVPR_2024/EmbodiedPose/sample_data/humor/for_embodied_demo/demo_clip_emb.pkl")