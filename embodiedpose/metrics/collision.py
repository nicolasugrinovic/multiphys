import numpy as np
from utils.stats import AverageMeter
from tqdm import tqdm


def compute_naturalness(data_pred, data_names, data_valid, static_scenes, device, data_is_norm=True, verbose=False,
                        data_name='prox', return_details=False, contact_threshold=0.01):
    '''
    verts is a list of 218 with tensors of (1, 64, 10475, 3)
    '''

    # few = ''
    # fpath = f'data/sample_data_names{few}.pkl'
    # write_pickle(fpath, data_names)
    # fpath = f'data/sample_static_scenes{few}.pkl'
    # write_pickle(fpath, static_scenes)
    # if verbose:

    non_coll_all = []
    contact_all = []
    contact_wa_all = []
    meter_non_coll, meter_con = AverageMeter('pve', ':6.6f'), AverageMeter('pve_wo_trans', ':6.6f')
    meter_con_wa = AverageMeter('pve', ':6.6f')
    meter_con_sDiff = AverageMeter('pve', ':6.6f')
    pbar = tqdm(range(len(data_pred)), disable=not verbose)
    for n in pbar:
        body_verts_batch, _, _, *_ = data_pred[n]
        for nn in range(len(body_verts_batch)):
            valid = data_valid[n][nn]
            name = data_names[n][nn]
            verts_hat = body_verts_batch[nn].cuda()
            cam_extrinsic = static_scenes[name]['camera_extrinsic'].cuda()
            grid_max = static_scenes[name]['grid_max'].cuda()
            grid_min = static_scenes[name]['grid_min'].cuda()
            sdf = static_scenes[name]['sdf'].cuda()
            scene_y = static_scenes[name]['scene_y'].cuda()
            s_sdf_batch = torch.tensor(sdf, dtype=torch.float32, device=device).unsqueeze(0)

            if data_name == 'prox':
                if data_is_norm:
                    verts_hat_norm = unnormalize_verts(verts_hat[None], None, None, None, scene_y[None],
                                                       grid_min[None], grid_max[None], inspect=False)
                    norm_verts_batch = verts_hat_norm.squeeze(0)
                else:
                    verts_hat_tf = geometric.verts_transform(verts_hat, cam_extrinsic[None])[0]
                    norm_verts_batch = (verts_hat_tf - grid_min) / (grid_max - grid_min) * 2 - 1
            elif data_name == 'humanise':
                try:
                    norm_verts_batch = norm_verts_sdf(verts_hat[None], grid_min[None], grid_max[None])[0]
                except:
                    print(f'Error in evaluate_scene naturalness with: {name}')
                    continue
                # scene_pc_norm = norm_verts_sdf(scene_pc[None], grid_min[None], grid_max[None])
            else:
                raise NotImplementedError

            B, n_verts, _ = norm_verts_batch.shape
            s_sdf_batch_ = s_sdf_batch.expand(B, 1, -1, -1, -1)
            body_sdf_batch = F.grid_sample(s_sdf_batch_,
                                           norm_verts_batch[:, :, [2, 1, 0]].view(-1, n_verts, 1, 1, 3),
                                           padding_mode='border')
            if 0:
                import matplotlib.pylab as pl
                import trimesh

                idx = 0
                faces_path = 'data/smplx_faces.pkl'
                faces = read_pickle(faces_path)
                ipath = Path(f"./inspect_out/metrics/sdf/{name}")
                ipath.mkdir(parents=True, exist_ok=True)
                save_trimesh(verts_hat[idx], faces, ipath / Path(f'body_pred.obj'))
                save_trimesh(norm_verts_batch[idx], faces, ipath / Path(f'body_pred_norm.obj'))
                save_trimesh(verts_hat_norm[0, idx], faces, ipath / Path(f'body_pred_unnorm.obj'))
                n = 64
                le = 1
                x = np.linspace(-le, le, n)
                y = np.linspace(-le, le, n)
                z = np.linspace(-le, le, n)
                xv, yv, zv = np.meshgrid(x, y, z)
                grid = np.stack([xv, yv, zv], axis=-1)  # .reshape(-1, 3)
                pts = grid.reshape(-1, 3)
                # seems that prox SDF is inverted: sdfs>0 is inside and sdfs<0 is outside
                inside = sdf < 0
                outisde = sdf > 0
                occ_pts_in = grid[inside.cpu()]
                occ_pts = grid[outisde.cpu()]
                colors = pl.cm.jet(sdf.cpu().reshape([-1])).squeeze()
                sdf_pc = trimesh.PointCloud(vertices=pts, colors=colors)
                sdf_pc.export(str(ipath / Path('sdf_visu.ply')));
                save_pointcloud(occ_pts, str(ipath / Path('sdf_occ_out.ply')))
                save_pointcloud(occ_pts_in, str(ipath / Path('sdf_occ_in.ply')))
                scene_pc = static_scenes[name]['scene_pc']
                save_pointcloud(scene_pc, str(ipath / Path('scene_pc.ply')))
                # the grid min and max should already be translated with center near the origin
                scene_pc_norm = norm_verts_sdf(scene_pc[None], grid_min[None], grid_max[None])[0, 0]
                save_pointcloud(scene_pc_norm, str(ipath / Path('scene_pc_norm.ply')))
                verts_norm_simple = norm_verts_sdf(verts_hat[idx][None], grid_min[None], grid_max[None])[0, 0]
                save_trimesh(verts_norm_simple[idx], faces, ipath / Path(f'verts_norm_simple.obj'))

            non_coll_s_batch = 0
            cont_score_batch = 0
            contact_score_wang_batch = 0
            contact_score_sceDiff_batch = 0
            for i in range(len(body_sdf_batch)):
                valid_i = valid[i]
                if not valid_i:
                    continue
                contact_score = torch.tensor(1.0, dtype=torch.float32, device=device)
                # if the number of negative sdf entries is less than one
                if body_sdf_batch[i].lt(0).sum().item() < 1:
                    # when there is no penetration
                    sdf_penet_score = torch.tensor(10475, dtype=torch.float32, device=device)
                    contact_score = torch.tensor(0.0, dtype=torch.float32, device=device)
                else:
                    sdf_penet_score = (body_sdf_batch[i] > 0).sum()

                dist1 = body_sdf_batch[i].min()
                if torch.sum(dist1 < contact_threshold) > 0:
                    contact_score_wang = torch.tensor(1.0, dtype=torch.float32, device=device)
                else:
                    contact_score_wang = torch.tensor(0.0, dtype=torch.float32, device=device)

                if torch.sum(dist1 < 0.02) > 0:
                    contact_score_sceDiff = torch.tensor(1.0, dtype=torch.float32, device=device)
                else:
                    contact_score_sceDiff = torch.tensor(0.0, dtype=torch.float32, device=device)

                collision_score = sdf_penet_score.float() / 10475.0
                non_coll_s_batch += collision_score
                cont_score_batch += contact_score
                contact_score_wang_batch += contact_score_wang
                contact_score_sceDiff_batch += contact_score_sceDiff

            non_coll_s_batch /= B
            cont_score_batch /= B
            contact_score_wang_batch /= B
            contact_score_sceDiff_batch /= B

            # if err > 500:
            #     print(f'too high error with idx: {j}')
            #     continue

            meter_non_coll.update(non_coll_s_batch, B)
            meter_con.update(cont_score_batch, B)
            meter_con_wa.update(contact_score_wang_batch, B)
            meter_con_sDiff.update(contact_score_sceDiff_batch, B)

            non_coll_all.append(non_coll_s_batch.item())
            contact_all.append(cont_score_batch.item())
            contact_wa_all.append(contact_score_wang_batch.item())

            if verbose:
                msg = ' | Non-collision {meter_coll.avg:.4f} | Contact {meter_con.avg:.4f} | Contact wang {meter_con_wa.avg:.4f}'.format(
                    meter_coll=meter_non_coll, meter_con=meter_con, meter_con_wa=meter_con_wa)
                # pbar.write(msg)
                pbar.set_postfix_str(msg)

    scores = {
        'non_collision': 100*meter_non_coll.avg,
        'contact': 100*meter_con.avg,
        'contact_wang': 100*meter_con_wa.avg,
        'contact_sceDiffuser': 100*meter_con_sDiff.avg,
    }
    if return_details:
        scores['non_collision_all'] = non_coll_all
        scores['contact_all'] = contact_all
        scores['contact_wang_all'] = contact_wa_all

    return scores


if __name__ == "__main__":
    pass
