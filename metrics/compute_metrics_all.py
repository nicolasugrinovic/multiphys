from utils.misc import write_str_txt
from metrics.compute_metrics import compute_metrics_main
import socket

hostname = socket.gethostname()
import sys


def updata_dict(args, exp_n, res_dict, print_str_eval, sub_dir=None):
    if sub_dir is not None:
        res_dict[f'{args.model_type}-{exp_n}-{sub_dir}'] = print_str_eval
    else:
        res_dict[f'{args.model_type}-{exp_n}'] = print_str_eval
    return res_dict


def filter_dict(results, metric_type):
    keep_keys = {
        'pose_mp': ['ga_jmse', 'fa_jmse', 'pampjpe'],
        'pose': ['root_dist', 'pa_mpjpe', 'mpjpe', 'accel_dist', 'vel_dist'],
        'sdf': ['penet_sdf'],
        'phys': ['skate', 'pentration', 'float'],
    }

    new_dict = {}
    for k in results:
        if k in keep_keys[metric_type]:
            new_dict[k] = results[k]
    return new_dict


def dict_to_csv(avg_dict, out_path_res, sep=","):
    """
    This takes a dict of dicts containing the scalar metrics.
    The first dict is the model name, the second dict is the metric name and the value.
    """
    CSV = ""
    for kk in avg_dict:
        # kk = 'baseline-normal_op'
        subCSV = f"{kk}"
        subHead = []
        for k, v in avg_dict[kk].items():
            subCSV += f"{sep}{v}"
            subHead.append(k)
        subCSV += f"\n"
        CSV += subCSV
    HEAD = f"Model{sep}" + f"{sep}".join(subHead)
    # CSV = f"\n{HEAD}\n{CSV}"
    CSV = f"{HEAD}\n{CSV}"

    if sep == ",":
        out_path_res_csv = out_path_res.replace(".txt", ".csv")
    elif sep == "\t":
        out_path_res_csv = out_path_res.replace(".txt", f".tsv")
    else:
        raise NotImplementedError

    with open(out_path_res_csv, "w") as file:
        file.write(CSV)
    print(f"Saving CSV to {out_path_res_csv}")


def main(args):
    RES_ROOT = f"./results/scene+/tcn_voxel_4_5_chi3d_multi_hum/results"
    path = f"{RES_ROOT}/{args.data_name}"

    if args.metric_type == 'sdf':
        out_path_res = f"{path}/eval_metrics_all-{args.metric_type}-{args.threshold}.txt"
    else:
        out_path_res = f"{path}/eval_metrics_all-{args.metric_type}.txt"

    if args.log:
        log_file = out_path_res.replace(".txt", ".log")
        f = open(log_file, 'w')
        sys.stdout = f

    model_types_all = ['slahmr', 'baseline', 'glamr']

    exp_names_all = [
        # 'normal_op',
        'slahmr_override_loop2',
    ]

    if args.data_name == 'expi':
        exp_names_all = [
            # 'normal_op',
            'slahmr_override_loop2',
        ]
    elif args.data_name == 'hi4d':
        exp_names_all = [
            # 'normal_op',
            'slahmr_override_loop2',
        ]

    res_dict = {}
    keys_dict = {}
    avg_dict = {}
    for exp_n in exp_names_all:
        args.model_type = model_types_all[1]
        args.exp_name = exp_n
        print_str_eval, seqs_list, results = compute_metrics_main(args, args.model_type, exp_names_all)
        res_dict = updata_dict(args, exp_n, res_dict, print_str_eval)
        keys_dict = updata_dict(args, exp_n, keys_dict, seqs_list)
        results = filter_dict(results, args.metric_type)
        avg_dict = updata_dict(args, exp_n, avg_dict, results)

        try:
            # if there is slahmr results on this dir then compute metrics
            args.model_type = model_types_all[0]
            args.exp_name = exp_n
            print_str_eval, seqs_list, results = compute_metrics_main(args, args.model_type, exp_names_all)
            res_dict = updata_dict(args, exp_n, res_dict, print_str_eval, args.sub_dir)
            keys_dict = updata_dict(args, exp_n, keys_dict, seqs_list, args.sub_dir)
            results = filter_dict(results, args.metric_type)
            avg_dict = updata_dict(args, exp_n, avg_dict, results)
        except:
            # print(f"NO SLAHMR RESULTS ON {exp_n}")
            pass

    final_str = ""
    for k, v in keys_dict.items():
        final_str += f"\n{k}:  \n {v}"
    print(final_str)

    print(f"COMPUTED: {len(seqs_list)} sequences")

    final_str = ""
    for k, v in res_dict.items():
        final_str += f"\n{k}:  \n {v}"
    print(final_str)

    print(f"**Saving eval metrics to {out_path_res}")
    write_str_txt(final_str, out_path_res)

    dict_to_csv(avg_dict, out_path_res, sep="\t")

    if args.log:
        f.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', type=int, choices=[0, 1], default=0)
    parser.add_argument('--model_type', type=str, choices=['slahmr', 'baseline'], default='baseline')
    parser.add_argument('--data_name', type=str, choices=['chi3d', 'hi4d', 'expi'], default='hi4d')
    parser.add_argument('--exp_name', type=str, choices=['normal_op', 'slahmr_override', 'overwrite_gt'],
                        default='normal_op')
    parser.add_argument('--filter_seq', type=str, default=None)
    parser.add_argument('--verbose', type=int, choices=[0, 1], default=0)
    parser.add_argument('--use_wbpos', type=int, choices=[0, 1], default=0)
    parser.add_argument('--use_smpl_gt', type=int, choices=[0, 1], default=0)
    parser.add_argument('--metric_type', type=str, default='pose_mp', choices=['pose_mp', 'sdf', 'phys'])
    parser.add_argument('--threshold', type=float, default=0.1, help="threshold for sdf loss")
    parser.add_argument("--sub_dir", type=str, default=None)
    parser.add_argument("--sla_exp_name", type=str, default=None)
    parser.add_argument('--log', type=int, choices=[0, 1], default=0)

    args = parser.parse_args()
    main(args)
