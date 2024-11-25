import numpy as np
from uhc.smpllib.smpl_eval import compute_metrics
from uhc.smpllib.smpl_eval import compute_phys_metrics
from utils.misc import read_pickle
from utils.misc import write_str_txt
from tqdm import tqdm
from metrics.collision import compute_penetration
from metrics.sdf_collision import compute_collision_sdf
from pathlib import Path
from metrics.sequence_lists import get_sequences_intersection_v2
from metrics.metrics_mp import compute_metrics_mp
import socket
hostname = socket.gethostname()
from collections import defaultdict
import pandas as pd
from metrics.prepare_slahmr_results import get_emb_paths


"""
PYTHONUNBUFFERED=1;
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia:/home/nugrinovic/.mujoco/mujoco210/bin;
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so;
DISPLAY=localhost:10.0
"""

seq_nums = ["s02", "s03", "s04"]

def format_results(results, ignore_keys=None):
    if ignore_keys is not None:
        results = {k: v for k, v in results.items() if k not in ignore_keys}

    res_avg = {k: np.mean(v) for k, v in results.items()}
    print_str_eval = "\t".join([f"{k}" for k, v in res_avg.items()])
    print_str_eval += "\n"
    print_str_eval += "\t".join([f"{v:.6f}" for k, v in res_avg.items()])
    print(print_str_eval)
    return res_avg, print_str_eval


def format_results_line(results, filter_keys=None, ignore_keys=None):
    if filter_keys is not None:
        results = {k: v for k, v in results.items() if k in filter_keys}
    if ignore_keys is not None:
        results = {k: v for k, v in results.items() if k not in ignore_keys}
    res_avg = {k: np.mean(v) for k, v in results.items()}
    print_str_eval = "".join([f"{k} {v:.6f} | " for k, v in res_avg.items()])
    # print(print_str_eval)
    return res_avg, print_str_eval


def save_per_frame_metrics(keep, metric_res_seq, out_path_csv_seq):
    """
    keep: list of keys to keep
    metric_res_seq: list of dicts, each dict is a metric result for one person containing a numpy array with dim (n_frames,)
    """
    metric_seq = {k: np.array([m[k] for m in metric_res_seq]).mean(0) for k in keep}
    if 'accel_dist' in keep:
        metric_seq['vel_dist'] = np.pad(metric_seq['vel_dist'], (1, 0), 'constant', constant_values=0)
        metric_seq['accel_dist'] = np.pad(metric_seq['accel_dist'], (2, 0), 'constant', constant_values=0)
    Path(out_path_csv_seq).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_seq).to_csv(out_path_csv_seq)


def from_dict_of_lists_to_list_of_dicts(dict_of_lists):
    list_of_dicts = [{}, {}]
    for key in dict_of_lists:
        pass
        val = dict_of_lists[key]
        for ni in range(val.shape[1]):
            list_of_dicts[ni][key] = val[:, ni]
    return list_of_dicts


def compute_metrics_main(args, model_type='baseline', exp_names_all=None):

    if exp_names_all is None:
        exp_names_all = [
                        'normal_op', # baseline
                         'slahmr_override_loop2',
                         ]

    if args.data_name == 'expi':
        exp_names_all = [
            'normal_op',
            'slahmr_override_loop2',
        ]
    elif args.data_name == 'hi4d':
        exp_names_all = [
            'normal_op',
            'slahmr_override_loop2',
        ]


    print(f"Computing metrics for {model_type}...")

    path, SEQS, ROOT, RES_ROOT = get_emb_paths(args.data_name, args.exp_name)

    met_n = f"{args.metric_type}_{args.threshold}" if args.metric_type != 'phys' else ""
    if args.metric_type == 'phys':
        out_path_penet = f"{path}/metrics/{model_type}/eval_metrics_PHYS_{model_type}-{args.exp_name}-{met_n}.txt"
    else:
        out_path_penet = f"{path}/metrics/{model_type}/eval_metrics_PENET_{model_type}-{args.exp_name}-{met_n}.txt"
    out_path_res = f"{path}/metrics/{model_type}/eval_metrics_{model_type}-{args.exp_name}-{args.metric_type}.txt"
    out_path_csv = f"{path}/metrics/{model_type}/eval_metrics_{model_type}-{args.exp_name}-{args.metric_type}.csv"
    results = defaultdict(list)

    # loop over all sequences
    for seq in SEQS:
        seqs_list = get_sequences_intersection_v2(args, RES_ROOT, seq, exp_names_all)

        print(f" Doing {seq}...")
        if model_type== 'baseline':
            file = f"{path}/{seq}.pkl"
        elif model_type== 'slahmr':
            file = f"{path}/{seq}_slahmr.pkl"
        else:
            raise ValueError(f"Unknown data_name: {model_type}")

        if args.metric_type=='phys':
            file = file.replace(".pkl", "_verts.pkl")

        print("reading results...")
        # res_data is a dict with seq names as keys, each dict element is a list of results for each person
        # results for each person is a dict with pred and gt keys:
        # 'gt', 'target', 'pred', 'gt_jpos', 'pred_jpos', 'world_body_pos', 'world_trans', 'percent', 'fail_safe',
        # 'percent', 'fail_safe', should be 1.0 and False respectively
        print(f"Reading file from: {file}")
        res_data = read_pickle(file)
        # we dont want default dicts
        res_data = dict(res_data)
        seq_keys = list(res_data.keys())
        data_keys = list(res_data[seq_keys[0]][0].keys())
        print(f"DATA KEYS are: {data_keys}")

        # go over each seq
        pbar = tqdm(res_data)
        if args.filter_seq is not None:
            pbar = tqdm([args.filter_seq])
            print("*** WARNING: filtering seqs")

        cnt = 0
        for key in pbar:
            if key not in seqs_list:
                tqdm.write(f"Warning: skipping seq {key}, not in the list!")
                continue

            exp_str = f" {args.exp_name} - {args.data_name}  -  {key} - "
            cnt += 1
            # go over each person
            smpl_data, pose_data, metric_res_seq = [], [], []
            # iter per person
            for n in range(len(res_data[key])):
                key_data = res_data[key][n]
                if args.metric_type == 'pose':
                    metric_res = compute_metrics(key_data, None, key, args, n, use_wbpos=args.use_wbpos,
                                                 use_smpl_gt=args.use_smpl_gt)
                    metric_res_seq.append(metric_res)
                    # per sequence average
                    metric_res = {k: np.mean(v) for k, v in metric_res.items() if k not in ['pa_mpjpe_per_jt', 'mpjpe_per_jt']}
                    for met_key in metric_res:
                        results[met_key].append(metric_res[met_key])
                    results['0_seq_name'].append(key)
                elif args.metric_type == 'pose_mp':
                    pass
                elif args.metric_type == 'phys':
                    metric_res = compute_phys_metrics(key_data, None, key, args, n, use_wbpos=args.use_wbpos,
                                                      use_smpl_gt=args.use_smpl_gt)
                    metric_res_seq.append(metric_res)
                    metric_res = {k: np.mean(v) for k, v in metric_res.items() if k not in ['pa_mpjpe_per_jt', 'mpjpe_per_jt']}
                    for met_key in metric_res:
                        results[met_key].append(metric_res[met_key])
                    results['0_seq_name'].append(key)
                try:
                    smpl_data.append(res_data[key][n]['smpl_pred'])
                    p_dict = {'pred_jpos': res_data[key][n]['pred_jpos'], 'gt_jpos': res_data[key][n]['gt_jpos']}
                    if 'gt_jpos_vis' in res_data[key][n]:
                        p_dict['gt_jpos_vis'] = res_data[key][n]['gt_jpos_vis']
                    pose_data.append(p_dict)
                except:
                    print(f"no smpl_pred for {key}! skipping...")
                    continue

            if args.metric_type == 'verts':
                mean_penet, mean_penet_gt = compute_penetration(smpl_data, verbose=args.verbose, debug=args.debug)
                results['penet_perc'].append(np.round(100*mean_penet, 4))
                results['0_seq_name'].append(key)
            elif args.metric_type == 'sdf':
                # this computes the cumulative sdf penetration for all frames for one sequence,
                # i.e., the sum of the penet per person for the whole sequence, it is not per frame, it is per sequence
                mean_penet = compute_collision_sdf(smpl_data, key, args, n, verbose=args.verbose, threshold=args.threshold)
                # mean_penet contains penet info for one sequence
                results['penet_sdf'].append(mean_penet)
                results['0_seq_name'].append(key)
            elif args.metric_type == 'pose':
                keep = ['root_dist', 'pa_mpjpe', 'mpjpe', 'mpjpe_g', 'accel_dist', 'vel_dist']
                out_path_csv_seq = Path(out_path_csv).parent / f"per_frame/{key}.csv"
                save_per_frame_metrics(keep, metric_res_seq, out_path_csv_seq)

                # keep = ['jpos_pred', 'jpos_gt']
                keep = ['pa_mpjpe_per_jt', 'mpjpe_per_jt']
                metric_seq = [{f"{k}_{n}": d[k] for k in keep} for n, d in enumerate(metric_res_seq)]
                new_dict = metric_seq[0]
                new_dict.update(metric_seq[1])
                for name in new_dict:
                    out_path_csv_seq = Path(out_path_csv).parent / f"per_frame_jts/{key}-{name}.csv"
                    Path(out_path_csv_seq).parent.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(new_dict[name].T).to_csv(out_path_csv_seq)

            elif args.metric_type == 'phys':
                pass

            elif args.metric_type == 'pose_mp':
                # have to compute pose_mp here because we need both persons at the same time
                metric_res = compute_metrics_mp(pose_data, None, key, args, n, use_wbpos=args.use_wbpos,
                                                use_smpl_gt=args.use_smpl_gt)
                metric_res_seq.append(metric_res)
                keep = ['ga_jmse_per_fr', 'fa_jmse_per_fr', 'pampjpe_per_fr']
                metric_res_fr = {k: v.cpu().numpy() for k, v in metric_res.items() if k in keep}
                metric_res_fr = from_dict_of_lists_to_list_of_dicts(metric_res_fr)

                out_path_csv_seq = Path(out_path_csv).parent / f"per_frame_mp/{key}.csv"
                save_per_frame_metrics(keep, metric_res_fr, out_path_csv_seq)

                # per sequence average
                for met_key in metric_res:
                    if met_key in keep:
                        continue
                    results[met_key].append(metric_res[met_key])
                results['0_seq_name'].append(key)


            else:
                raise NotImplementedError

            if args.metric_type == 'pose':
                res_avg, print_str_eval = format_results_line(results, filter_keys=["pa_mpjpe", "mpjpe"])
                pbar.set_description(exp_str + print_str_eval)
            else:
                res_avg, print_str_eval = format_results_line(results, ignore_keys=["0_seq_name"])
                pbar.set_description(exp_str + print_str_eval)


        # save at each subject, in case of crash
        if args.metric_type == 'verts' or args.metric_type == 'sdf':
            if 'pose' in args.metric_type:
                res_avg, print_str_eval = format_results_line(results, ignore_keys=["0_seq_name"])
                print(f"**Saving eval metrics to {out_path_res}")
                write_str_txt(print_str_eval, out_path_res)
            else:
                res_avg, print_str_eval = format_results_line(results, ignore_keys=["0_seq_name"])
                print(f"**Saving eval metrics to {out_path_penet}")
                write_str_txt(print_str_eval, out_path_penet)
                pbar.set_description(exp_str + print_str_eval)

            pd.DataFrame(results).to_csv(out_path_csv)


    # compute final average
    if 'pose' in args.metric_type:
        res_avg, print_str_eval = format_results(results, ignore_keys=["0_seq_name", "mpjpe_g",
                                                                       "succ", "jpos_pred", "jpos_gt",
                                                                       "pred_sum_check", "gt_sum_check",
                                                                       "n_frames"])
        # save results
        print(f"\n**Saving eval metrics to {out_path_res}")
        write_str_txt(print_str_eval, out_path_res)
    else:
        res_avg, print_str_eval = format_results(results, ignore_keys=["0_seq_name"])
        print(f"**Saving eval metrics to {out_path_penet}")
        write_str_txt(print_str_eval, out_path_penet)

    pd.DataFrame(results).to_csv(out_path_csv)

    return print_str_eval, seqs_list, res_avg



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', type=int, choices=[0, 1], default=0)
    parser.add_argument('--model_type', type=str, choices=['slahmr', 'baseline', 'glamr'], default='baseline')
    parser.add_argument('--data_name', type=str, choices=['chi3d', 'hi4d', 'expi'], default='hi4d')
    parser.add_argument('--exp_name', type=str, default='normal_op')
    parser.add_argument('--filter_seq', type=str, default=None)
    parser.add_argument('--verbose', type=int, choices=[0, 1], default=0)
    parser.add_argument('--use_wbpos', type=int, choices=[0, 1], default=0)
    parser.add_argument('--use_smpl_gt', type=int, choices=[0, 1], default=0)
    parser.add_argument('--metric_type', type=str, default='verts',
                        choices=['verts', 'volume', 'pose', 'pose_mp', 'sdf', 'phys'])
    parser.add_argument('--threshold', type=float, default=0.1)
    parser.add_argument("--sla_exp_name", type=str, default=None)
    args = parser.parse_args()

    compute_metrics_main(args, model_type=args.model_type)
