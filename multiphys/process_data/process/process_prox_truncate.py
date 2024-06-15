import numpy as np
from pathlib import Path
from utils.misc import read_pickle
from utils.misc import read_pickle_compatible
from utils.misc import write_pickle
import joblib

def trim_data():
    excluded = [
        "betas", 
        "seq_name", 
        "gender",
        "cam",
        "points3d",
    ]
    path = "sample_data/thirdeye_anns_prox_overlap_no_clip.pkl"
    out_path = "sample_data/thirdeye_anns_prox_only_one.pkl"
    data = joblib.load(path)

    # curr_data --> keys: (['pose_aa', 'pose_6d', 'pose_body', 'trans', 'trans_vel', 'root_orient',
    # 'root_orient_vel', 'joints', 'joints_vel', 'betas', 'seq_name', 'gender', 'joints2d', 'points3d', 'cam'])
    curr_data = data["N0Sofa_00145_01"]
    trimmed_data = {}
    for k,v in curr_data.items():
        if k not in excluded:
            new_v = v[0:100]
            trimmed_data[k] = new_v
            # print(k, new_v.shape)

    curr_data.update(trimmed_data)
    new_data = {"N0Sofa_00145_01": curr_data}
    joblib.dump(new_data, out_path)


def truncate_all_but_first(truncate_jts2d=False):
    """
    Truncate all but the first frame of each sequence.
    """
    excluded = [
        "betas",
        "seq_name",
        "gender",
        "cam",
        "points3d",
    ]
    path = "sample_data/thirdeye_anns_prox_overlap_no_clip.pkl"
    out_path = "sample_data/thirdeye_anns_prox_only_one_trunc_j2dtrunc.pkl" if truncate_jts2d else "sample_data/thirdeye_anns_prox_only_one_trunc.pkl"
    data = joblib.load(path)

    new_len = 300

    # curr_data --> keys: (['pose_aa', 'pose_6d', 'pose_body', 'trans', 'trans_vel', 'root_orient',
    # 'root_orient_vel', 'joints', 'joints_vel', 'betas', 'seq_name', 'gender', 'joints2d', 'points3d', 'cam'])
    curr_data = data["N0Sofa_00145_01"]
    trimmed_data = {}
    for k,v in curr_data.items():
        if k not in excluded:
            new_v = np.zeros_like(v[0:new_len])
            if k=="joints2d":
                if truncate_jts2d:
                    # also truncate those OP joints and keep DEKR joints + OP nose
                    new_v = np.zeros_like(v[0:new_len])
                    new_v[:, :15] = v[0:new_len, :15]
                else:
                    new_v = v[0:new_len]
            # elif k=="points3d":
            #     curr_data[k] = None
            else:
                new_v[0] = v[0]
            trimmed_data[k] = new_v
            # print(k, new_v.shape)

    curr_data.update(trimmed_data)
    new_data = {"N0Sofa_00145_01": curr_data}
    joblib.dump(new_data, out_path)
    print(f"Saved to {out_path}")



if __name__ == "__main__":
    # trim_data()
    truncate_all_but_first()
    # truncate_all_but_first(truncate_jts2d=True)
