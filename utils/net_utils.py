import socket
import os
from pathlib import Path

hostname = socket.gethostname()

def get_hostname():
    import socket
    hostname = socket.gethostname()
    return hostname

def replace_orion_root(path):
    is_pathlib = isinstance(path, Path)
    if is_pathlib:
        path = str(path)
    orion_root = "/sailhome/nugrinov"
    if "mnt/Data" in path:
        kada_root = "/mnt/Data/nugrinovic"
    else:
        kada_root = "/home/nugrinovic"
    if "oriong" in hostname:
        path = path.replace(kada_root, orion_root)
    if is_pathlib:
        path = Path(path)

    return path


def replace_slahmr_path(path):
    # path_root = "/home/nugrinovic/code/NEURIPS_2023"
    orion_root = "/sailhome/nugrinov/code/CVPR_2024/slahmr_release"

    if "mnt/Data" in path:
        kada_root = "/mnt/Data/nugrinoviccode/NEURIPS_2023"
    else:
        kada_root = "/home/nugrinovic/code/NEURIPS_2023"

    ext_path = path.split(kada_root)[-1]
    if "oriong" in hostname:
        opath = os.path.join(orion_root, ext_path[1:])
    else:
        opath = path
    return opath


def replace_emb_path(path):
    orion_root = "/sailhome/nugrinov"
    if "mnt/Data" in path:
        kada_root = "/mnt/Data/nugrinovic"
    else:
        kada_root = "/home/nugrinovic"

    ext_path = path.split(kada_root)[-1]
    if "oriong" in hostname:
        opath = os.path.join(orion_root, ext_path[1:])
    else:
        opath = path
    return opath