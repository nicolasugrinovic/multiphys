import numpy as np

def to_numpy(cam):
    for k, v, in cam.items():
        is_np = isinstance(v, np.ndarray)
        if is_np or v is None:
            continue
        else:
            cam[k] = v.cpu().numpy()
    return cam


