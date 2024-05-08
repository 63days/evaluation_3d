import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

import h5py
import jutils
import numpy as np
import torch
from pytorch_lightning import seed_everything
from tqdm import tqdm
import torch

from ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from PyTorchEMD.emd import earth_mover_distance as EMD
from utils import seed_everything, to_unit_cube, to_unit_sphere, load_pc_from_mesh_direc

"""
############ Metric Functions ################
"""
cham3D = chamfer_3DDist()

def _pairwise_EMD_CD_(sample_pcs, ref_pcs, batch_size, accelerated_cd=True):
    N_sample = sample_pcs.shape[0]
    N_ref = ref_pcs.shape[0]
    all_cd = []
    all_emd = []
    iterator = range(N_sample)
    for sample_b_start in tqdm(iterator):
        sample_batch = sample_pcs[sample_b_start]

        cd_lst = []
        emd_lst = []
        for ref_b_start in range(0, N_ref, batch_size):
            ref_b_end = min(N_ref, ref_b_start + batch_size)
            ref_batch = ref_pcs[ref_b_start:ref_b_end]

            batch_size_ref = ref_batch.size(0)
            sample_batch_exp = sample_batch.view(1, -1, 3).expand(batch_size_ref, -1, -1)
            sample_batch_exp = sample_batch_exp.contiguous()

            dl, dr, _, _ = cham3D(sample_batch_exp.cuda(), ref_batch.cuda())
            cd_lst.append((dl.mean(dim=1) + dr.mean(dim=1)).view(1, -1).detach().cpu())

            emd_batch = EMD(sample_batch_exp.cuda(), ref_batch.cuda(), transpose=False)
            emd_lst.append(emd_batch.view(1, -1).detach().cpu())

        cd_lst = torch.cat(cd_lst, dim=1)
        emd_lst = torch.cat(emd_lst, dim=1)
        all_cd.append(cd_lst)
        all_emd.append(emd_lst)

    all_cd = torch.cat(all_cd, dim=0)  # N_sample, N_ref
    all_emd = torch.cat(all_emd, dim=0)  # N_sample, N_ref

    return all_cd, all_emd


def lgan_mmd_cov(all_dist):
    N_sample, N_ref = all_dist.size(0), all_dist.size(1)
    min_val_fromsmp, min_idx = torch.min(all_dist, dim=1)
    min_val, _ = torch.min(all_dist, dim=0)
    mmd = min_val.mean()
    mmd_smp = min_val_fromsmp.mean()
    cov = float(min_idx.unique().view(-1).size(0)) / float(N_ref)
    cov = torch.tensor(cov).to(all_dist)
    return {
        'mmd': mmd,
        'cov': cov,
        # 'lgan_mmd_smp': mmd_smp,
    }


# Adapted from https://github.com/xuqiantong/GAN-Metrics/blob/master/framework/metric.py
def knn(Mxx, Mxy, Myy, k, sqrt=False):
    """
    Cut off matrices so that they become a square matrix.
    """
    num_x = Mxx.shape[0]
    num_y = Myy.shape[0]
    min_len = min(num_x, num_y)
    Mxx = Mxx[:min_len, :min_len]
    Mxy = Mxy[:min_len, :min_len]
    Myy = Myy[:min_len, :min_len]
    print(f"Mxx: {Mxx.shape} Mxy: {Mxy.shape} Myy: {Myy.shape}")
    # ========================

    n0 = Mxx.size(0)
    n1 = Myy.size(0)
    label = torch.cat((torch.ones(n0), torch.zeros(n1))).to(Mxx)
    M = torch.cat((torch.cat((Mxx, Mxy), 1), torch.cat((Mxy.transpose(0, 1), Myy), 1)), 0)
    if sqrt:
        M = M.abs().sqrt()
    INFINITY = float('inf')
    val, idx = (M + torch.diag(INFINITY * torch.ones(n0 + n1).to(Mxx))).topk(k, 0, False)

    count = torch.zeros(n0 + n1).to(Mxx)
    for i in range(0, k):
        count = count + label.index_select(0, idx[i])
    pred = torch.ge(count, (float(k) / 2) * torch.ones(n0 + n1).to(Mxx)).float()

    s = {
        'tp': (pred * label).sum(),
        'fp': (pred * (1 - label)).sum(),
        'fn': ((1 - pred) * label).sum(),
        'tn': ((1 - pred) * (1 - label)).sum(),
    }

    s.update({
        'precision': s['tp'] / (s['tp'] + s['fp'] + 1e-10),
        'recall': s['tp'] / (s['tp'] + s['fn'] + 1e-10),
        'acc_t': s['tp'] / (s['tp'] + s['fn'] + 1e-10),
        'acc_f': s['tn'] / (s['tn'] + s['fp'] + 1e-10),
        'acc': torch.eq(label, pred).float().mean(),
    })
    return s
""" ############################################## """

def compute_all_metrics(sample_pcs, ref_pcs, batch_size):
    def print_results(res):
        for k, v in res.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if "mmd-CD" in k:
                print(f"{k}: {v * 1000} x 10^-3")
            if "mmd-EMD" in k:
                print(f"{k}: {v * 100} x 10^-2")
            if "cov" in k or "1-NN" in k:
                print(f"{k}: {v * 100} %")

    results = {}
    M_rs_cd, M_rs_emd = _pairwise_EMD_CD_(ref_pcs, sample_pcs, batch_size)

    res_cd = lgan_mmd_cov(M_rs_cd.t())
    results.update({
        "%s-CD" % k: v for k, v in res_cd.items()
    })

    res_emd = lgan_mmd_cov(M_rs_emd.t())
    results.update({
        "%s-EMD" % k: v for k, v in res_emd.items()
    })
    print_results(results)

    print("Compute RR.......")
    M_rr_cd, M_rr_emd = _pairwise_EMD_CD_(ref_pcs, ref_pcs, batch_size)
    
    print("Compute SS.......")
    M_ss_cd, M_ss_emd = _pairwise_EMD_CD_(sample_pcs, sample_pcs, batch_size)
    
    # 1-NN results
    one_nn_results = {}

    one_nn_cd_res = knn(M_rr_cd, M_rs_cd, M_ss_cd, 1, sqrt=False)
    one_nn_results.update({
        "1-NN-CD-%s" % k: v for k, v in one_nn_cd_res.items() if 'acc' in k
    })
    one_nn_emd_res = knn(M_rr_emd, M_rs_emd, M_ss_emd, 1, sqrt=False)
    one_nn_results.update({
        "1-NN-EMD-%s" % k: v for k, v in one_nn_emd_res.items() if 'acc' in k
    })
    print_results(one_nn_results)

    results.update(one_nn_results)

    return results

def main(args):
    now = datetime.now().strftime("%m-%d-%H%M%S")
    seed = 63
    seed_everything(seed)
    print(f"[*] SEED FIXED: {seed}")

    """
    Directories
    """
    fake_dir = Path(args.fake_dir)
    ref_dir = Path(args.ref_dir)
    if args.output_dir is None:
        output_dir = fake_dir / f"{now}"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    output_save_path = output_dir / f"eval_results.csv"

    fake_pc = load_pc_from_mesh_direc(fake_dir).cuda()
    ref_pc = load_pc_from_mesh_direc(ref_dir).cuda()
    
    if args.normalize == "unit_cube":
        normalize_func = to_unit_cube
    elif args.normalize == "unit_sphere":
        normalize_func = to_unit_sphere
    else:
        normalize_func = lambda x: x
    fake_pc , ref_pc = list(map(lambda x: normalize_func(x), [fake_pc, ref_pc]))
    if args.normalize != "none":
        print("[*] Normalize point clouds within {args.normalize_func}.")


    eval_results = compute_all_metrics(fake_pc, ref_pc, batch_size=512)

    with open(output_save_path, "w") as f:
        w = csv.DictWriter(f, eval_results.keys())
        w.writeHeader()
        w.writerow(eval_results)
    print(f"[*] Output saved at: {output_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake_dir", type=str, help="directory storing generated samples (.obj)")
    parser.add_argument("--ref_dir", type=str, help="directory storing reference samples (.obj)")
    parser.add_argument("--output_dir", type=str, default=None, help="directory to store outputs. if None, it will be set {fake_dir}/{command_run_time}")
    parser.add_argument("--normalize", type=str, choices=["none", "unit_cube", "unit_sphere"], default="unit_cube")

    args = parser.parse_args()

    main(args)
