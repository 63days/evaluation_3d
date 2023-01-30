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


class Evaluator:
    def __init__(self, gt_set=None, pred_set=None, batch_size=64, device="cuda:0"):
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
        
        if gt_set is not None:
            self.update_gt(gt_set)
        if pred_set is not None:
            self.update_pred(pred_set)
        if gt_set is not None and pred_set is not None:
            self.check_data()

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
    
    def compute_all_metrics(self, gt_set=None, pred_set=None, batch_size=None, device=None, verbose=False, return_distance_matrix=False):
        """
        Output:
            MMD-CD
            COV-CD
            1-NNA-CD
        """
        results = {}

        gt_set = gt_set if gt_set is not None else self.gt_set
        pred_set = pred_set if pred_set is not None else self.pred_set

        Mxy = self.compute_chamfer_distance(gt_set, pred_set, batch_size, device, verbose=verbose)
        if verbose:
            print("[*] Finished computing (pred, gt) distance")
        Mxx = self.compute_chamfer_distance(gt_set, gt_set, batch_size, device, verbose=verbose)
        if verbose:
            print("[*] Finished computing (gt, gt) distance")
        Myy = self.compute_chamfer_distance(pred_set, pred_set, batch_size, device, verbose=verbose)
        if verbose:
            print("[*] Finished computing (pred, pred) distance")
        
        out = self.compute_mmd_and_coverage(Mxy, device)
        if verbose:
            for k, v in out.items():
                print(f"{k}: {v}")
        results.update(out) # mmd and coverage
        
        nna_1 = self.compute_1_nna(Mxx, Mxy, Myy)
        results["1-NNA-CD"] = nna_1
        if verbose:
            print("1-NNA-CD:", nna_1)

        if return_distance_matrix:
            results["distance_matrix_xx"] = Mxx
            results["distance_matrix_xy"] = Mxy
            results["distance_matrix_yy"] = Myy

        return results
        
    def compute_chamfer_distance(self, A, B, batch_size=None, device=None, verbose=False):
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
            batchA = nputil.np2th(A[i:i+1]).repeat(batch_size, 1, 1).to(device) #[batch_size,M,D]
            
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
                    batchB = torch.cat([batchB, torch.zeros(padding_size, *batchB.shape[1:]).float().to(device)], 0)

                batchcd = (p3d_chamfer_distance(batchA, batchB, batch_reduction=None, point_reduction="mean")[0].cpu().numpy())
                # print(b_sidx, b_eidx, batchB.shape, dist_mat[i,b_sidx:b_eidx].shape, batchcd.shape, batchcd[b_sidx:b_eidx].shape)
                dist_mat[i,b_sidx:b_eidx] = batchcd[:original_num_B]
                if verbose:
                    pbarB.set_description(f"Dist: {batchcd[:original_num_B].mean()}")
                batchB = None
                # sysutil.clean_gpu()

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
        min_val_fromsmp, min_idx = torch.min(Mxy, dim=1) # min dist for each prediction
        min_val, _ = torch.min(Mxy, dim=0) # min dist for each GT
        mmd = min_val.mean()
        mmd_smp = min_val_fromsmp.mean()
        cov = float(min_idx.unique().view(-1).size(0)) / float(N_gt)
        return {
                "MMD-CD": mmd.item(),
                "COV-CD": cov,
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
        # device = device if device is not None else self.device
        device = "cpu" # cpu computation would be enough.

        Mxx, Mxy, Myy = tuple(map(lambda x: nputil.np2th(x).to(device), [Mxx, Mxy, Myy]))
        label = torch.cat((torch.ones(n0), torch.zeros(n1))).to(Mxx)
        """
        """
        M = torch.cat(
            [torch.cat([Mxx, Mxy], 1), torch.cat([Mxy.transpose(0, 1), Myy], 1)], 0
        )
        if sqrt:
            M = M.abs().sqrt()
        INFINITY = float("inf")
        val, idx = (M + torch.diag(INFINITY * torch.ones(n0 + n1).to(Mxx))).topk(k, 0, False)
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


if __name__ == "__main__":
    dummy_gt = torch.randn([512, 16,3])
    dummy_pred = torch.randn([512, 16,3])

    evaluator = Evaluator(dummy_gt, dummy_pred, 128, device="cuda:0")

    evaluator.compute_all_metrics(verbose=True)
