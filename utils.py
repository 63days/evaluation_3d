import os
import multiprocessing
import open3d as o3d
from pathlib import Path
import numpy as np
import torch

def poisson_sampling(vert: np.array, face: np.array, num_points=2048):
    vert_o3d = o3d.utility.Vector3dVector(vert)
    face_o3d = o3d.utility.Vector3iVector(face)
    mesh_o3d = o3d.geometry.TriangleMesh(vert_o3d, face_o3d)
    pc_o3d = mesh_o3d.sample_points_poisson_disk(num_points)
    pc = np.asarray(pc_o3d.points).astype(np.float32)
    return pc

def seed_everything(seed=2024):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def to_unit_cube(verts: torch.Tensor):
    """
    verts: [B,N,3]
    """
    max_vals = verts.max(1, keepdim=True)[0] #[B,1,3]
    min_vals = verts.min(1,keepdim=True)[0] #[B,1,3]
    max_range = (max_vals - min_vals).max(-1)[0] / 2 #[B,1]
    center = (max_vals + min_vals) / 2 #[B,1,3]
    
    verts = verts - center
    verts = verts / max_range[..., None]
    
    return verts

def to_unit_sphere(verts: torch.Tensor):
    """
    verts: [B,N,3]
    """
    mean = verts.mean(1, keepdim=True)
    verts = verts - mean
    scale = torch.max(torch.sqrt(torch.sum(verts**2, -1, keepdim=True)), 1, keepdim=True)[0]
    verts = verts / scale
    
    return verts

def load_obj(name: str):
    verts = []
    faces = []
    with open(name, "r") as f:
        lines = [line.rstrip() for line in f]

        for line in lines:
            if line.startswith("v "):
                verts.append(np.float32(line.split()[1:4]))
            elif line.startswith("f "):
                faces.append(np.int32([item.split("/")[0] for item in line.split()[1:4]]))

        v = np.vstack(verts)
        f = np.vstack(faces) - 1
        return v, f


def load_pc_from_mesh_direc(direc: os.PathLike):
    direc = Path(direc)
    pc_save_path = direc / f"o3d_all_pointclouds.pt"
    if pc_save_path.exists():
        print(f"[*] Load existing point cloud data from `{pc_save_path}`")
        return torch.load(pc_save_path, map_location="cpu")

    print(f"[*] Start loading meshes and sampling point clouds")
    
    file_paths = [f for f in direc.glob("*.obj")]
    file_paths = sorted(file_paths)
    
    def work_func(path):
        v, f = load_obj(path)
        pc = poisson_sampling(v, f)
        return pc

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        pc_list = pool.map(work_func, file_paths)
    
    pc_list = np.stack(pc_list, axis=0) #[B,2048,3]
    pc_list = torch.from_numpy(pc_list).float()
    torch.save(pc_list, pc_save_path)
    print(f"[*] Saved point clouds at {pc_save_path}")

    return pc_list

