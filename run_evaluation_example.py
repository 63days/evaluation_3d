import torch
import argparse
import numpy as np
from eval3d import Evaluator
from pathlib import Path
import h5py
import jutils
from datetime import datetime


def unit_sphere_normal_batch(pcs):
    offset = pcs.mean(1, keepdims=True)
    pcs = pcs - offset
    scale = np.max(np.sqrt(np.sum(pcs**2, -1)), 1)
    pcs = pcs / scale[..., None, None]
    return pcs


def eval_func(data_top_dir, metric=["emd", "chamfer"], num_shapes=1000, num_samples=2048, mix=False, seed=63):
    now = datetime.now().strftime("%m-%d-%H%M%S")

    if isinstance(data_top_dir, str):
        data_top_dir = Path(data_top_dir)
    device = "cuda:0"
    batch_size = 512
    num_shapes = num_shapes
    num_samples = num_samples

    gt_pc_path = "/home/juil/projects/3D_CRISPR/data/shapenet_groundtruth/all_pointclouds.hdf5"
    gt_pc_path = "/scratch/juil/iccv2023/data/shapenet_groundtruth/all_pointclouds.hdf5"
    with h5py.File(gt_pc_path) as f:
        gt_pc = f["data"][:].astype(np.float32)

    with h5py.File(data_top_dir / "all_pointclouds.hdf5") as f:
        pred_pc = f["data"][:].astype(np.float32)

    np.random.seed(seed)
    if mix:
        np.random.shuffle(gt_pc)
        np.random.shuffle(pred_pc)

    gt_pc = gt_pc[:num_shapes, :num_samples]
    pred_pc = pred_pc[:num_shapes, :num_samples]
    assert (
        len(gt_pc) >= num_shapes
        and len(pred_pc) >= num_shapes
        and gt_pc.shape[1] >= num_samples
        and pred_pc.shape[1] >= num_samples
    )
    print(f"[*] Finished loading pointcloud data.")

    gt_pc = unit_sphere_normal_batch(gt_pc)
    pred_pc = unit_sphere_normal_batch(pred_pc)
    
    if isinstance(metric, str):
        metric_list = [metric]
    elif isinstance(metric, list) or isinstance(metric, tuple):
        metric_list = metric
    else:
        raise AssertionError()

    for metric in metric_list:
        evaluator = Evaluator(
            gt_set=gt_pc, pred_set=pred_pc, metric=metric, batch_size=batch_size
        )

        eval_res = evaluator.compute_all_metrics(verbose=True, return_distance_matrix=True)
        eval_res_wo_dist_mat = {k: v for k, v in eval_res.items() if "distance_matrix" not in k}
        dist_mats = {k : v for k, v in eval_res.items() if "distance_matrix" in k}

        with open(data_top_dir / f"eval_res-{metric}.txt", "a") as f:
            f.write(f"{now}\n")
            for k, v in eval_res_wo_dist_mat.items():
                f.write(f"{k}: {v}\n")
        
        for k, v in dist_mats.items():
            np.save(data_top_dir / f"{k}-{metric}-{now}.npy", v)
        

if __name__ == "__main__":
    # DATA_TOP_DIR = Path("/home/juil/projects/3D_CRISPR/data/spaghetti_official")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_top_dir", type=str)
    parser.add_argument("--metric", nargs="+")
    parser.add_argument("--seed", type=int, default=63)
    parser.add_argument("--mix", action="store_true", default=False)
    args = parser.parse_args()

    DATA_TOP_DIR = Path(args.data_top_dir)
    assert DATA_TOP_DIR.exists()
    eval_func(DATA_TOP_DIR, metric=args.metric, mix=args.mix, seed=args.seed)
