"""
From https://github.com/stevenygd/PointFlow/tree/master/metrics
"""
import torch
import numpy as np
import warnings
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from tqdm.auto import tqdm
from jutils import nputil, thutil, sysutil
from pytorch3d.loss import chamfer_distance as p3d_chamfer_distance


class Evaluator:
    def __init__(self, gt_set, pred_set, batch_size=64, device="cuda:0"):
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
        assert (
            len(gt_set.shape) == len(pred_set.shape) == 3
        ), f"gt_set: {gt_set.shape} | pred_set: {pred_set.shape}. Shapes of both sets should be [NUM_SHAPES, NUM_POINTS, DIM]."

        self.num_gt, self.num_points_gt, self.dim_gt = gt_set.shape
        self.num_pred, self.num_points_pred, self.dim_pred = pred_set.shape

        assert (
            self.dim_gt == self.dim_pred
        ), f"Dimensions of two sets should be the same."
    
    def compute_all_metrics(self, gt_set=None, pred_set=None, batch_size=None, device=None):
        results = {}

        gt_set = gt_set if gt_set is not None else self.gt_set
        pred_set = pred_set if pred_set is not None else self.pred_set

        Mxy = self.compute_chamfer_distance(gt_set, pred_set, batch_size, device)
        Mxx = self.compute_chamfer_distance(gt_set, gt_set, batch_size, device)
        Myy = self.compute_chamfer_distance(pred_set, pred_set, batch_size, device)
        
        out = self.compute_mmd_and_coverage(Mxy, device)
        for k, v in out.items():
            print(f"{k}: {v}")
        results.update(out) # mmd and coverage
        
        nna_1 = self.compute_1_nna(Mxx, Mxy, Myy, device)
        results["1-NNA-CD"] = nna_1
        print("1-NNA-CD:", nna_1)

        return results
        
    def compute_chamfer_distance(self, A, B, batch_size=None, device=None):
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

        num_batches_A = compute_num_batches(N1)
        num_batches_B = compute_num_batches(N2)

        for i in range(num_batches_A):
            a_sidx = i * batch_size
            a_eidx = a_sidx + batch_size
            batchA = nputil.np2th(A[a_sidx:a_eidx]).to(device)
            for j in range(num_batches_B):
                b_sidx = j * batch_size
                b_eidx = b_sidx + batch_size
                batchB = nputil.np2th(B[b_sidx:b_eidx]).to(device)

                batchcd = (
                    p3d_chamfer_distance(
                        batchA, batchB, batch_reduction=None, point_reduction="mean"
                    )
                    .cpu()
                    .numpy()
                )
                dist_mat[a_sidx:a_eidx, b_sidx:b_eidx] = batchcd

                batchB = None
                sysutil.clean_gpu()  # for GPU memory efficiency.

        return dist_mat

    def compute_earth_movers_distance(self, A, B, _batch_size=None, _device=None):
        # TODO: Implement it.
        print("Not implemented yet.")
    
    def compute_mmd_and_coverage(self, Mxy, device=None):
        """
        Input:
            Mxy: [N1, N2] np.ndarray or torch.Tensor
            x and y should be prediction and ground truth, respectively.
        Output:
            dict=(MMD: float, COV: float)
        """
        Mxy = nputil.np2th(Mxy).to(device)
        N_pred, N_gt = Mxy.shape[:2]
        min_val_fromsmp, min_idx = torch.min(Mxy, dim=1)
        min_val, _ = torch.min(Mxy, dim=0)
        mmd = min_val.mean()
        mmd_smp = min_val_fromsmp.mean()
        cov = float(min_idx.unique().view(-1).size(0)) / float(N_gt)
        return {
                "MMD-CD": mmd.item(),
                "COV-CD": cov,
                }

    def compute_1_nna(self, Mxx, Mxy, Myy, device=None):
        """
        Input:
            Mxx: [N1,N1] np.ndarray or torch.Tensor
            Mxy: [N1,N2] np.ndarray or torch.Tensor
            Myy: [N2,N2] np.ndarray or torch.Tensor
        Output:
            acc: float 
        """
        return self.compute_knn(Mxx, Mxy, Myy, k=1, device=device, return_only_knn_acc=True)

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
        # device = device if device is not None else self.device
        device = "cpu" # cpu computation would be enough.

        Mxx, Mxy, Myy = tuple(map(lambda x: x.to(device), [Mxx, Mxy, Myy]))
        label = torch.cat((torch.ones(n0), torch.zeros(n1))).to(Mxx)
        """
        """
        M = torch.cat(
            [torch.cat([Mxx, Mxy], 1), torch.cat([Mxy.transpose(0, 1), Myy], 1)], 0
        )
        if sqrt:
            M = M.abs().sqrt()

        INFINITY = float("inf")
        val, idx = (M + torch.diag(INFINITY * torch.ones(n0 + n1).to(Mxx))).topk(k, 0)
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


# Adapted from https://github.com/xuqiantong/
# GAN-Metrics/blob/master/framework/metric.py
# k=1 -> 1-NN. The lower, the better.
def knn(Mxx, Mxy, Myy, k, sqrt=False):
    n0 = Mxx.size(0)
    n1 = Myy.size(0)
    label = torch.cat((torch.ones(n0), torch.zeros(n1))).to(Mxx)
    M = torch.cat(
        [torch.cat((Mxx, Mxy), 1), torch.cat((Mxy.transpose(0, 1), Myy), 1)], 0
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
        "tp": (pred * label).sum(),
        "fp": (pred * (1 - label)).sum(),
        "fn": ((1 - pred) * label).sum(),
        "tn": ((1 - pred) * (1 - label)).sum(),
    }

    s.update(
        {
            "precision": s["tp"] / (s["tp"] + s["fp"] + 1e-10),
            "recall": s["tp"] / (s["tp"] + s["fn"] + 1e-10),
            "acc_t": s["tp"] / (s["tp"] + s["fn"] + 1e-10),
            "acc_f": s["tn"] / (s["tn"] + s["fp"] + 1e-10),
            "acc": torch.eq(label, pred).float().mean(),
        }
    )
    return s


def lgan_mmd_cov(all_dist):
    N_sample, N_ref = all_dist.size(0), all_dist.size(1)
    min_val_fromsmp, min_idx = torch.min(all_dist, dim=1)
    min_val, _ = torch.min(all_dist, dim=0)
    mmd = min_val.mean()
    mmd_smp = min_val_fromsmp.mean()
    cov = float(min_idx.unique().view(-1).size(0)) / float(N_ref)
    cov = torch.tensor(cov).to(all_dist)
    return {
        "lgan_mmd": mmd,
        "lgan_cov": cov,
        "lgan_mmd_smp": mmd_smp,
    }


def lgan_mmd_cov_match(all_dist):
    N_sample, N_ref = all_dist.size(0), all_dist.size(1)
    min_val_fromsmp, min_idx = torch.min(all_dist, dim=1)
    min_val, _ = torch.min(all_dist, dim=0)
    mmd = min_val.mean()
    mmd_smp = min_val_fromsmp.mean()
    cov = float(min_idx.unique().view(-1).size(0)) / float(N_ref)
    cov = torch.tensor(cov).to(all_dist)
    return {
        "lgan_mmd": mmd,
        "lgan_cov": cov,
        "lgan_mmd_smp": mmd_smp,
    }, min_idx.view(-1)


def compute_all_metrics(sample_pcs, ref_pcs, batch_size):
    results = {}

    print("Pairwise EMD CD")
    M_rs_cd, M_rs_emd = _pairwise_EMD_CD_(ref_pcs, sample_pcs, batch_size)

    ## CD
    res_cd = lgan_mmd_cov(M_rs_cd.t())
    results.update({"%s-CD" % k: v for k, v in res_cd.items()})

    ## EMD
    # res_emd = lgan_mmd_cov(M_rs_emd.t())
    # results.update({
    #     "%s-EMD" % k: v for k, v in res_emd.items()
    # })

    for k, v in results.items():
        print("[%s] %.8f" % (k, v.item()))

    M_rr_cd, M_rr_emd = _pairwise_EMD_CD_(ref_pcs, ref_pcs, batch_size)
    M_ss_cd, M_ss_emd = _pairwise_EMD_CD_(sample_pcs, sample_pcs, batch_size)

    # 1-NN results
    ## CD
    one_nn_cd_res = knn(M_rr_cd, M_rs_cd, M_ss_cd, 1, sqrt=False)
    results.update(
        {"1-NN-CD-%s" % k: v for k, v in one_nn_cd_res.items() if "acc" in k}
    )
    ## EMD
    # one_nn_emd_res = knn(M_rr_emd, M_rs_emd, M_ss_emd, 1, sqrt=False)
    # results.update({
    #     "1-NN-EMD-%s" % k: v for k, v in one_nn_emd_res.items() if 'acc' in k
    # })

    return results


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


_EMD_NOT_IMPL_WARNED = False
def emd_approx(sample, ref):
    global _EMD_NOT_IMPL_WARNED
    emd = torch.zeros([sample.size(0)]).to(sample)
    if not _EMD_NOT_IMPL_WARNED:
        _EMD_NOT_IMPL_WARNED = True
        print("\n\n[WARNING]")
        print("  * EMD is not implemented due to GPU compatability issue.")
        print("  * We will set all EMD to zero by default.")
        print(
            "  * You may implement your own EMD in the function `emd_approx` in ./evaluation/evaluation_metrics.py"
        )
        print("\n")
    return emd


def EMD_CD(sample_pcs, ref_pcs, batch_size, reduced=False):
    N_sample = sample_pcs.shape[0]
    N_ref = ref_pcs.shape[0]
    assert N_sample == N_ref, "REF:%d SMP:%d" % (N_ref, N_sample)

    cd_lst = []
    emd_lst = []
    iterator = range(0, N_sample, batch_size)

    for b_start in tqdm(iterator, desc="EMD-CD"):
        b_end = min(N_sample, b_start + batch_size)
        sample_batch = sample_pcs[b_start:b_end]
        ref_batch = ref_pcs[b_start:b_end]

        dl, dr = distChamfer(sample_batch, ref_batch)
        cd_lst.append(dl.mean(dim=1) + dr.mean(dim=1))

        emd_batch = emd_approx(sample_batch, ref_batch)
        emd_lst.append(emd_batch)

    if reduced:
        cd = torch.cat(cd_lst).mean()
        emd = torch.cat(emd_lst).mean()
    else:
        cd = torch.cat(cd_lst)
        emd = torch.cat(emd_lst)

    results = {
        "MMD-CD": cd,
        "MMD-EMD": emd,
    }
    return results


def _pairwise_EMD_CD_(sample_pcs, ref_pcs, batch_size, verbose=True):
    N_sample = sample_pcs.shape[0]
    N_ref = ref_pcs.shape[0]
    all_cd = []
    all_emd = []
    iterator = range(N_sample)
    if verbose:
        iterator = tqdm(iterator, desc="Pairwise EMD-CD")
    for sample_b_start in iterator:
        sample_batch = sample_pcs[sample_b_start]

        cd_lst = []
        emd_lst = []
        sub_iterator = range(0, N_ref, batch_size)
        # if verbose:
        #     sub_iterator = tqdm(sub_iterator, leave=False)
        for ref_b_start in sub_iterator:
            ref_b_end = min(N_ref, ref_b_start + batch_size)
            ref_batch = ref_pcs[ref_b_start:ref_b_end]

            batch_size_ref = ref_batch.size(0)
            point_dim = ref_batch.size(2)
            sample_batch_exp = sample_batch.view(1, -1, point_dim).expand(
                batch_size_ref, -1, -1
            )
            sample_batch_exp = sample_batch_exp.contiguous()

            dl, dr = distChamfer(sample_batch_exp, ref_batch)
            cd_lst.append((dl.mean(dim=1) + dr.mean(dim=1)).view(1, -1))

            emd_batch = emd_approx(sample_batch_exp, ref_batch)
            emd_lst.append(emd_batch.view(1, -1))

        cd_lst = torch.cat(cd_lst, dim=1)
        emd_lst = torch.cat(emd_lst, dim=1)
        all_cd.append(cd_lst)
        all_emd.append(emd_lst)

    all_cd = torch.cat(all_cd, dim=0)  # N_sample, N_ref
    all_emd = torch.cat(all_emd, dim=0)  # N_sample, N_ref

    return all_cd, all_emd


# Borrow from https://github.com/ThibaultGROUEIX/AtlasNet
def distChamfer(a, b):
    x, y = a, b
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind = torch.arange(0, num_points).to(a).long()
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = rx.transpose(2, 1) + ry - 2 * zz
    return P.min(1)[0], P.min(2)[0]


if __name__ == "__main__":
    a = torch.randn([16, 2048, 3]).cuda()
    b = torch.randn([16, 2048, 3]).cuda()
    print(EMD_CD(a, b, batch_size=8))
