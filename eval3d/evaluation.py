"""
From https://github.com/stevenygd/PointFlow/tree/master/metrics
"""
import torch
from pytorch3d.loss import chamfer_distance as p3d_chamfer_distance
import numpy as np
import warnings
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from tqdm import tqdm

"""
jutils: https://github.com/63days/jutils
"""
from jutils import nputil, thutil, sysutil

from eval3d import emdModule


class Evaluator:
    def __init__(
        self,
        gt_set=None,
        pred_set=None,
        batch_size=64,
        device="cuda:0",
        metric="chamfer",
    ):
        """
        gt_set: [num_gt, num_points, dim]
        pred_set: [num_pred, num_points, dim]
        batch_size: int
        device: default = cuda:0
        """
        self.gt_set = gt_set
        self.pred_set = pred_set
        self.batch_size = batch_size  # when measuring distances.
        self.device = device
        self.metric = metric
        assert metric in [
            "chamfer",
            "l2",
            "emd",
        ], f"metric should be chamfer, l2 or emd. Input: {metric}."
        print(f"[!] Evaluator metric is defined as {metric}.")

        if gt_set is not None:
            self.update_gt(gt_set)
        if pred_set is not None:
            self.update_pred(pred_set)
        if gt_set is not None and pred_set is not None:
            self.check_data()

    def compute_pairwise_distance(
        self, A, B, batch_size=None, device=None, verbose=False
    ):
        if self.metric == "chamfer":
            return self.compute_chamfer_distance(
                A, B, batch_size=batch_size, device=device, verbose=verbose
            )
        elif self.metric == "l2":
            return self.compute_l2_distance(A, B)
        elif self.metric == "emd":
            return self.compute_earth_movers_distance(
                A, B, batch_size=batch_size, device=device, verbose=verbose
            )
        else:
            raise AssertionError()

    def update_gt(self, gt):
        self.gt_set = gt

    def update_pred(self, pred):
        self.pred_set = pred

    def check_data(self):
        gt_set = self.gt_set
        pred_set = self.pred_set
        assert (
            len(gt_set.shape) == len(pred_set.shape) == 3
        ), f"gt_set: {gt_set.shape} | pred_set: {pred_set.shape}. Shapes of both sets should be [NUM_SHAPES, NUM_POINTS, DIM]."

        self.num_gt, self.num_points_gt, self.dim_gt = gt_set.shape
        self.num_pred, self.num_points_pred, self.dim_pred = pred_set.shape

        assert (
            self.dim_gt == self.dim_pred
        ), f"Dimensions of two sets should be the same."

    def compute_all_metrics(
        self,
        gt_set=None,
        pred_set=None,
        batch_size=None,
        device=None,
        verbose=False,
        return_distance_matrix=False,
        compute_jsd_together=True,
    ):
        """
        Output:
            JSD
            MMD-{metric}
            COV-{metric}
            1-NNA-{metric}
        """
        results = {}

        gt_set = gt_set if gt_set is not None else self.gt_set
        pred_set = pred_set if pred_set is not None else self.pred_set
        
        if compute_jsd_together:
            jsd = self.compute_jsd(gt_set, pred_set)
            results["JSD"] = jsd
            if verbose:
                print(f"JSD: {jsd}")

        Mxy = self.compute_pairwise_distance(
            gt_set, pred_set, batch_size, device, verbose
        )
        if verbose:
            print("[*] Finished computing (gt, pred) distance")
        Mxx = self.compute_pairwise_distance(
            gt_set, gt_set, batch_size, device, verbose
        )
        if verbose:
            print("[*] Finished computing (gt, gt) distance")
        Myy = self.compute_pairwise_distance(
            pred_set, pred_set, batch_size, device, verbose
        )
        if verbose:
            print("[*] Finished computing (pred, pred) distance")

        out = self.compute_mmd_and_coverage(Mxy, device)
        if verbose:
            for k, v in out.items():
                print(f"{k}: {v}")
        results.update(out)  # mmd and coverage

        nna_1 = self.compute_1_nna(Mxx, Mxy, Myy)
        results[f"1-NNA-{self.metric}"] = nna_1
        if verbose:
            print(f"1-NNA-{self.metric}:", nna_1)

        if return_distance_matrix:
            results["distance_matrix_xx"] = Mxx
            results["distance_matrix_xy"] = Mxy
            results["distance_matrix_yy"] = Myy

        return results

    def compute_chamfer_distance(
        self, A, B, batch_size=None, device=None, verbose=False
    ):
        """
        Input:
            A: np.ndarray or torch.Tensor [N1,M,D]
            B: np.ndarray or torch.Tensor [N2,M,D]
            _batch_size: int (Optional)
        Output:
            dist_mat: np.ndarray [N1,N2]
        """
        N1, N2 = len(A), len(B)
        dist_mat = np.zeros((N1, N2))
        batch_size = int(batch_size) if batch_size is not None else self.batch_size
        device = device if device is not None else self.device

        compute_num_batches = lambda num: int(np.ceil(num / batch_size))

        num_batches_B = compute_num_batches(N2)
        """
        d(A1,B1), d(A1,B2), d(A1,B3), ...
        d(A2,B1), d(A2,B2), d(A2,B3), ...
        """
        pbarA = range(len(A))
        if verbose:
            pbarA = tqdm(pbarA, leave=False)
        for i in pbarA:
            if i % (len(pbarA) // 5) == 0 and i != 0:
                print(f"Computing {i} CD finished.")
            batchA = (
                nputil.np2th(A[i : i + 1]).repeat(batch_size, 1, 1).to(device)
            )  # [batch_size,M,D]

            pbarB = range(num_batches_B)
            if verbose:
                pbarB = tqdm(pbarB, leave=False)
            for j in pbarB:
                b_sidx = j * batch_size
                b_eidx = b_sidx + batch_size
                batchB = nputil.np2th(B[b_sidx:b_eidx]).to(device)
                original_num_B = len(batchB)
                if len(batchB) < batch_size:
                    padding_size = batch_size - len(batchB)
                    batchB = torch.cat(
                        [
                            batchB,
                            torch.zeros(padding_size, *batchB.shape[1:])
                            .float()
                            .to(device),
                        ],
                        0,
                    )

                batchcd = (
                    p3d_chamfer_distance(
                        batchA, batchB, batch_reduction=None, point_reduction="mean"
                    )[0]
                    .cpu()
                    .numpy()
                )
                dist_mat[i, b_sidx:b_eidx] = batchcd[:original_num_B]
                if verbose:
                    pbarB.set_description(f"Dist: {batchcd[:original_num_B].mean()}")
                batchB = None
                # sysutil.clean_gpu()

        return dist_mat

    def compute_earth_movers_distance(
        self, A, B, batch_size=None, device=None, verbose=False
    ):
        """
        Input:
            A: np.ndarray or torch.Tensor [N1,M,D]
            B: np.ndarray or torch.Tensor [N2,M,D]
            _batch_size: int (Optional)
        Output:
            dist_mat: np.ndarray [N1,N2]
        """
        emd = emdModule()
        N1, N2 = len(A), len(B)
        dist_mat = np.zeros((N1, N2))
        batch_size = int(batch_size) if batch_size is not None else self.batch_size
        device = device if device is not None else self.device

        compute_num_batches = lambda num: int(np.ceil(num / batch_size))

        num_batches_B = compute_num_batches(N2)
        """
        d(A1,B1), d(A1,B2), d(A1,B3), ...
        d(A2,B1), d(A2,B2), d(A2,B3), ...
        """
        pbarA = range(len(A))
        if verbose:
            pbarA = tqdm(pbarA, leave=False)
        for i in pbarA:
            if len(pbarA) // 5 != 0:
                if i % (len(pbarA) // 5) == 0 and i != 0:
                    print(f"Computing {i} EMD finished.")
            batchA = (
                nputil.np2th(A[i : i + 1]).repeat(batch_size, 1, 1).to(device)
            )  # [batch_size,M,D]

            pbarB = range(num_batches_B)
            if verbose:
                pbarB = tqdm(pbarB, leave=False)
            for j in pbarB:
                b_sidx = j * batch_size
                b_eidx = b_sidx + batch_size
                batchB = nputil.np2th(B[b_sidx:b_eidx]).to(device)
                original_num_B = len(batchB)
                if len(batchB) < batch_size:
                    padding_size = batch_size - len(batchB)
                    batchB = torch.cat(
                        [
                            batchB,
                            torch.zeros(padding_size, *batchB.shape[1:])
                            .float()
                            .to(device),
                        ],
                        0,
                    )

                dis, assignment = emd(batchA, batchB, 0.05, 3000)
                batch_emd = np.sqrt(thutil.th2np(dis)).mean(-1)
                dist_mat[i, b_sidx:b_eidx] = batch_emd[:original_num_B]
                if verbose:
                    pbarB.set_description(f"Dist: {batch_emd[:original_num_B].mean()}")
                batchB = None
                # sysutil.clean_gpu()

        return dist_mat

    def compute_l2_distance(self, A, B):
        """
        Input:
            A: np.ndarray or torch.Tensor [N1,M,D]
            B: np.ndarray or torch.Tensor [N2,M,D]
        Output:
            dist_mat: np.ndarray [N1,N2]
        """
        A = nputil.np2th(A)
        B = nputil.np2th(B)
        N1, M, D = A.shape
        N2 = B.shape[0]

        A = A.reshape(N1, M * D).unsqueeze(1)  # [N1,1,H]
        B = B.reshape(N2, M * D).unsqueeze(0)  # [1,N2,H]

        dist_mat = (A - B).norm(dim=-1)  # [N1,N2]
        dist_mat = thutil.th2np(dist_mat)
        return dist_mat

    def compute_mmd_and_coverage(self, M_gt_pred, device=None):
        """
        Input:
            M_gt_pred: [N1, N2] np.ndarray or torch.Tensor
            Each row and column should be ground truth and prediction, respectively.
        Output:
            dict=(MMD: float, COV: float)
        """
        Mxy = M_gt_pred
        Mxy = nputil.np2th(Mxy).to(device)
        N_gt, N_pred = Mxy.shape[:2]

        min_val_fromsmp, min_idx_fromsmp = torch.min(Mxy, dim=0)  # sample -> GT
        min_val, _ = torch.min(Mxy, dim=1)  # GT -> sample

        mmd = min_val.mean()
        cov = float(min_idx_fromsmp.unique().view(-1).size(0)) / float(N_gt)

        return {
            f"MMD-{self.metric}": mmd.item(),
            f"COV-{self.metric}": cov,
        }

    def compute_1_nna(self, Mxx, Mxy, Myy):
        """
        Input:
            Mxx: [N1,N1] np.ndarray or torch.Tensor
            Mxy: [N1,N2] np.ndarray or torch.Tensor
            Myy: [N2,N2] np.ndarray or torch.Tensor
        Output:
            acc: float
        """
        return self.compute_knn(Mxx, Mxy, Myy, k=1, return_only_knn_acc=True)

    def compute_knn(self, Mxx, Mxy, Myy, k=1, sqrt=False, return_only_knn_acc=True):
        """
        Input:
            Mxx: [N1,N1]
            Mxy: [N1,N2]
            Myy: [N2,N2]
            k: k-nearest-neighbors
            return_only_knn_acc=False: return dictionary.
        Output:
            precision, recall, acc_t, acc_f, acc
            (We will report acc.)
        """
        n0, n1 = len(Mxx), len(Myy)
        device = "cpu"  # cpu computation would be enough.

        Mxx, Mxy, Myy = tuple(
            map(lambda x: nputil.np2th(x).to(device), [Mxx, Mxy, Myy])
        )
        label = torch.cat((torch.ones(n0), torch.zeros(n1))).to(Mxx)
        """
        """
        M = torch.cat(
            [torch.cat([Mxx, Mxy], 1), torch.cat([Mxy.transpose(0, 1), Myy], 1)], 0
        )
        if sqrt:
            M = M.abs().sqrt()
        INFINITY = float("inf")
        val, idx = (M + torch.diag(INFINITY * torch.ones(n0 + n1).to(Mxx))).topk(
            k, 0, False
        )
        count = torch.zeros(n0 + n1).to(Mxx)
        for i in range(0, k):
            count = count + label.index_select(0, idx[i])
        pred = torch.ge(count, (float(k) / 2) * torch.ones(n0 + n1).to(Mxx)).float()

        s = {
            "tp": (pred * label).sum().item(),
            "fp": (pred * (1 - label)).sum().item(),
            "fn": ((1 - pred) * label).sum().item(),
            "tn": ((1 - pred) * (1 - label)).sum().item(),
        }
        s.update(
            {
                "precision": s["tp"] / (s["tp"] + s["fp"] + 1e-10),
                "recall": s["tp"] / (s["tp"] + s["fn"] + 1e-10),
                "acc_t": s["tp"] / (s["tp"] + s["fn"] + 1e-10),
                "acc_f": s["tn"] / (s["tn"] + s["fp"] + 1e-10),
                "acc": torch.eq(label, pred).float().mean().item(),
            }
        )
        if return_only_knn_acc:
            return s["acc"]
        return s

    def compute_jsd(self, gt_set=None, pred_set=None, resolution=28):
        """
        Input:
            gt_set: np.ndarray or torch.Tensor [N1, M1, 3]
            pred_set: np.ndarray or torch.Tensor [N2, M2, 3]
        Output:
            jsd: float    
        """
        gt_set = gt_set if gt_set is not None else self.gt_set
        pred_set = pred_set if pred_set is not None else self.pred_set
        gt_set = thutil.th2np(gt_set)
        pred_set = thutil.th2np(pred_set)
        return jsd_between_point_cloud_sets(sample_pcs=pred_set, ref_pcs=gt_set, resolution=resolution)

    
    

#######################################################
# JSD : from https://github.com/optas/latent_3d_points
#######################################################
def unit_cube_grid_point_cloud(resolution, clip_sphere=False):
    """Returns the center coordinates of each cell of a 3D grid with
    resolution^3 cells, that is placed in the unit-cube. If clip_sphere it True
    it drops the "corner" cells that lie outside the unit-sphere.
    """
    grid = np.ndarray((resolution, resolution, resolution, 3), np.float32)
    spacing = 1.0 / float(resolution - 1)
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                grid[i, j, k, 0] = i * spacing - 0.5
                grid[i, j, k, 1] = j * spacing - 0.5
                grid[i, j, k, 2] = k * spacing - 0.5

    if clip_sphere:
        grid = grid.reshape(-1, 3)
        grid = grid[norm(grid, axis=1) <= 0.5]

    return grid, spacing


def jsd_between_point_cloud_sets(sample_pcs, ref_pcs, resolution=28):
    """Computes the JSD between two sets of point-clouds,
       as introduced in the paper
    ```Learning Representations And Generative Models For 3D Point Clouds```.
    Args:
        sample_pcs: (np.ndarray S1xR2x3) S1 point-clouds, each of R1 points.
        ref_pcs: (np.ndarray S2xR2x3) S2 point-clouds, each of R2 points.
        resolution: (int) grid-resolution. Affects granularity of measurements.
    """
    in_unit_sphere = True
    sample_grid_var = entropy_of_occupancy_grid(sample_pcs, resolution, in_unit_sphere)[
        1
    ]
    ref_grid_var = entropy_of_occupancy_grid(ref_pcs, resolution, in_unit_sphere)[1]
    return jensen_shannon_divergence(sample_grid_var, ref_grid_var)


def entropy_of_occupancy_grid(pclouds, grid_resolution, in_sphere=False, verbose=False):
    """Given a collection of point-clouds, estimate the entropy of
    the random variables corresponding to occupancy-grid activation patterns.
    Inputs:
        pclouds: (numpy array) #point-clouds x points per point-cloud x 3
        grid_resolution (int) size of occupancy grid that will be used.
    """
    epsilon = 10e-4
    bound = 0.5 + epsilon
    if abs(np.max(pclouds)) > bound or abs(np.min(pclouds)) > bound:
        if verbose:
            warnings.warn("Point-clouds are not in unit cube.")

    if in_sphere and np.max(np.sqrt(np.sum(pclouds**2, axis=2))) > bound:
        if verbose:
            warnings.warn("Point-clouds are not in unit sphere.")

    grid_coordinates, _ = unit_cube_grid_point_cloud(grid_resolution, in_sphere)
    grid_coordinates = grid_coordinates.reshape(-1, 3)
    grid_counters = np.zeros(len(grid_coordinates))
    grid_bernoulli_rvars = np.zeros(len(grid_coordinates))
    nn = NearestNeighbors(n_neighbors=1).fit(grid_coordinates)

    for pc in tqdm(pclouds, desc="JSD"):
        _, indices = nn.kneighbors(pc)
        indices = np.squeeze(indices)
        for i in indices:
            grid_counters[i] += 1
        indices = np.unique(indices)
        for i in indices:
            grid_bernoulli_rvars[i] += 1

    acc_entropy = 0.0
    n = float(len(pclouds))
    for g in grid_bernoulli_rvars:
        if g > 0:
            p = float(g) / n
            acc_entropy += entropy([p, 1.0 - p])

    return acc_entropy / len(grid_counters), grid_counters


def jensen_shannon_divergence(P, Q):
    if np.any(P < 0) or np.any(Q < 0):
        raise ValueError("Negative values.")
    if len(P) != len(Q):
        raise ValueError("Non equal size.")

    P_ = P / np.sum(P)  # Ensure probabilities.
    Q_ = Q / np.sum(Q)

    e1 = entropy(P_, base=2)
    e2 = entropy(Q_, base=2)
    e_sum = entropy((P_ + Q_) / 2.0, base=2)
    res = e_sum - ((e1 + e2) / 2.0)

    res2 = _jsdiv(P_, Q_)

    if not np.allclose(res, res2, atol=10e-5, rtol=0):
        warnings.warn("Numerical values of two JSD methods don't agree.")

    return res


def _jsdiv(P, Q):
    """another way of computing JSD"""

    def _kldiv(A, B):
        a = A.copy()
        b = B.copy()
        idx = np.logical_and(a > 0, b > 0)
        a = a[idx]
        b = b[idx]
        return np.sum([v for v in a * np.log2(a / b)])

    P_ = P / np.sum(P)
    Q_ = Q / np.sum(Q)

    M = 0.5 * (P_ + Q_)

    return 0.5 * (_kldiv(P_, M) + _kldiv(Q_, M))


if __name__ == "__main__":
    """
    Two point clouds should have same size and be normalized to [0,1].
    The number of points should be a multiple of 1024.
    The batch size should be no greater than 512.
    """
    dummy_gt = torch.randn([32, 1024, 3])  
    dummy_pred = torch.randn([32, 1024, 3])

    evaluator = Evaluator(dummy_gt, dummy_pred, 128, device="cuda:0", metric="emd")
    evaluator.compute_all_metrics(verbose=True)

    evaluator = Evaluator(dummy_gt, dummy_pred, 128, device="cuda:0", metric="l2")
    evaluator.compute_all_metrics(verbose=True, compute_jsd_together=False)
