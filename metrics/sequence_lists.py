import numpy as np
import pandas as pd
from utils.misc import write_txt
from utils.video import get_file_path
from pathlib import Path


EMB_ROOT = "/sailhome/nugrinov/code/CVPR_2024/EmbodiedPose"
RES_ROOT = f"{EMB_ROOT}/results/scene+/tcn_voxel_4_5_chi3d_multi_hum/results"

def read_from_csv(args):
    path = f"{RES_ROOT}/{args.data_name}/slahmr_override/metrics/eval_metrics_baseline-slahmr_override.csv"
    df = pd.read_csv(path)
    df = df.drop(columns=['Unnamed: 0'])
    seqs = df['0_seq_name'].unique()
    return seqs

def get_sequences(res_root_norm, is_heuristic=False):
    """"
    read sequences from resutlsl dir, however the sequence folder might not exist if the sequence failed
     and was terminated, therefore check results.pkl
     """
    if is_heuristic:
        exp_name = res_root_norm.parts[-2]
        # pad heuristics results with original results as heuristics can be a subset of the original results
        orig_exp = exp_name.split("_optim_naive_heuristic")[0]
        res_root_norm = Path(str(res_root_norm).replace(exp_name, orig_exp))
    seqs_norm = sorted(res_root_norm.glob('*'))
    seqs_norm = [seq.stem for seq in seqs_norm if seq.is_dir()]
    seqs_norm = [seq for seq in seqs_norm if get_file_path(res_root_norm, seq, "results.pkl").is_file()]
    # seqs_norm = [seq.stem for seq in seqs_norm if seq/'results.pkl' in seq.glob('*')]
    return seqs_norm

def get_sequences_intersection(res_root_norm, res_root_over):
    seqs_norm = get_sequences(res_root_norm)
    seqs_over = get_sequences(res_root_over)
    seqs = sorted(set(seqs_norm) & set(seqs_over))
    return seqs


def get_sequences_intersection_v2(args, RES_ROOT, seq, exp_names_all):

    print(f"\nCOMMON SEQ list, reading from : {RES_ROOT}/{args.data_name} | seq: {seq}")
    print(f"\nEXPERIMENTS list is : {exp_names_all}\n")

    roots = []
    sequences = []
    for exp_n in exp_names_all:
        res_root = Path(f"{RES_ROOT}/{args.data_name}/{exp_n}/{seq}")
        is_heuristic = 'heuristic' in exp_n
        seqs_norm = get_sequences(res_root, is_heuristic=is_heuristic)
        roots.append(res_root)
        sequences.append(seqs_norm)

    seqs = [set(c) for c in sequences]
    lens_str = [f"{exp_names_all[n]} : {len(c)}" for n, c in enumerate(seqs)]
    seqs_norm = seqs[0]
    for i in range(1, len(seqs)):
        seqs_norm &= seqs[i]
    print()
    print("FOUND:")
    for str_l in lens_str:
        print(f" {str_l} ")
    print()

    print(f"\nCOMMON SEQUENCE list: {seqs_norm}")
    print(f"number of seqs: {len(seqs_norm)}\n")

    return seqs_norm

def main(args):
    res_root_norm = Path(f"{RES_ROOT}/{args.data_name}/normal_op")
    res_root_over = Path(f"{RES_ROOT}/{args.data_name}/slahmr_override")
    # read folder names
    seqs = get_sequences_intersection(res_root_norm, res_root_over)

    # seqs = read_from_csv(args)
    outp = f"{EMB_ROOT}/metrics/seqs_list_for_metrics_{args.data_name}.txt"
    write_txt(outp, seqs)
    pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', type=int, choices=[0, 1], default=0)
    parser.add_argument('--data_name', type=str, choices=['chi3d', 'hi4d', 'expi'], default='hi4d')
    args = parser.parse_args()

    main(args)
