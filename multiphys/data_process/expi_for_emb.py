import csv
import numpy as np
from tqdm import tqdm
from pathlib import Path
from utils.misc import write_pickle
from utils.expi_util import read_calib

ACRO1_CAMS = [
    'cam20',
    'cam30',
]
ACRO2_CAMS = [
    'cam19',
    'cam37',
]

CAMS = {'acro1': ACRO1_CAMS, 'acro2': ACRO2_CAMS}


def parse_cam_data(cam_data):
    intrinsics = cam_data['K'].astype(np.float32)
    R = cam_data['R'].astype(np.float32)
    T = cam_data['T'].T.astype(np.float32)
    # this is for visu only
    focal = [intrinsics[0, 0], intrinsics[1, 1]]
    center = [cam_data['xc'], cam_data['yc']]
    K = {'f': focal, 'c': center}
    cam_dict = {'R': R, 'T': T, 'K': K}
    return cam_dict


def read_tsv_data(root_path, offset=0):
    tsv_file = open(root_path+'mocap_cleaned.tsv')
    read_tsv = csv.reader(tsv_file, delimiter=",")
    gt = {}
    t = 1
    for row in read_tsv:
        if t >= 2:
            img_id = t-2+offset
            img_name = 'img-' + str(img_id).zfill(6) + '.jpg'
            gt[img_name] = np.array([float(g) for g in row]).reshape((36, 3))
        t += 1
    tsv_file.close()
    return gt


def read_gt_clean(root_path):
    # read data with img name
    align_file = open(root_path+'talign.csv')
    read_align = csv.reader(align_file, delimiter=",")
    t = 1
    for row in read_align:
        if t == 1:
            off = [r for r in row]
            if off[0] == 'mesh.start':
                offset = int(off[1])
            else:
                print('ERROR: error in getting mesh.start')
        t += 1
    ## read gt
    gt = read_tsv_data(root_path, offset=0)
    return gt


def main():
    data_name = 'expi'
    DATA_ROOT = Path(f"path/to/expi/dataset")
    OUTPUT_DIR = "./data/expi"
    SUBJECTS = ["acro1", "acro2"]
    for subj_name in SUBJECTS:
        imgs_root = DATA_ROOT / subj_name
        sequences = sorted(imgs_root.glob('*'))
        # filter "video" folder
        sequences = [subj for subj in sequences if subj.is_dir()]

        data_dict_p1 = {}
        data_dict_p2 = {}
        act_cnt = 0
        succ_cnt = 0
        for n, seq_name in enumerate(tqdm(sequences)):
            for expi_cam in CAMS[subj_name]:
                act_cnt += 1
                try:
                    data_path = str(seq_name)+"/"
                    data_dict = read_gt_clean(str(seq_name)+"/")
                except:
                    try:
                        # print("WARNING: align file not found, reading joints directly")
                        data_dict = read_tsv_data(data_path, offset=0)
                    except:
                        # print(f"ERROR: problem with GT data for {seq_name}")
                        continue

                cam_path = seq_name / 'calib-new.xml'
                # expi images are 2048 × 2048
                try:
                    cam_num = int(expi_cam[-2:])
                    cam_data = read_calib(str(cam_path), cam_num)
                except:
                    print(f"ERROR: problem with cam data for {seq_name}")
                    continue
                cam_dict = parse_cam_data(cam_data)

                joints_3d_or = np.stack(list(data_dict.values()))
                joints_3d = joints_3d_or.reshape([-1, 2, 18, 3])
                joints_3d = joints_3d.transpose([1, 0, 2, 3])
                succ_cnt +=1
                emb_data_p1 = {'joints_3d': joints_3d[0] / 1000.}
                emb_data_p2 = {'joints_3d': joints_3d[1] / 1000.}
                emb_data_p1["cam2world"] = cam_dict
                emb_data_p2["cam2world"] = cam_dict
                act = seq_name.stem
                data_dict_p1[f"{subj_name}_{act}_{expi_cam}"] = emb_data_p1
                data_dict_p2[f"{subj_name}_{act}_{expi_cam}"] = emb_data_p2

        keys = list(data_dict_p2.keys())
        print(keys)
        out_path = OUTPUT_DIR / Path(f'{data_name}_{subj_name}_embodied_cam2w_p1.pkl')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_pickle(data_dict_p1, out_path)

        out_path = OUTPUT_DIR / Path(f'{data_name}_{subj_name}_embodied_cam2w_p2.pkl')
        write_pickle(data_dict_p2, out_path)

        data_keys = list(data_dict_p1[keys[0]].keys())
        print(f"DATA KEYS are {data_keys}")

        print(f"saved to {out_path}")
        print(f"{subj_name}: Total actions: {act_cnt}, successfully read : {succ_cnt} ")


if __name__ == "__main__":
    main()
