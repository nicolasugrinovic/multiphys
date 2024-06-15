import numpy as np
from pathlib import Path
import os
import subprocess

def main(args):
    path = "/sailhome/nugrinov/code/CVPR_2024/EmbodiedPose/results/scene+/tcn_voxel_4_5_chi3d_multi_hum/results/expi/normal_op"
    folders = Path(path).glob("*")
    # filter
    folders = [f for f in folders if 'acro' not in str(f)]
    for folder in folders:

        cmd = f"mv {folder} {path}/acro1/acro1_{folder.stem}"
        subprocess.call(cmd, shell=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default="chi3d")
    args = parser.parse_args()

    main(args)
