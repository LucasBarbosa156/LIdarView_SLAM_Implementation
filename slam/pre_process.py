import numpy as np
import open3d as o3d


def preprocess(pcd_path, voxel_size=0.1):
    pcd = o3d.io.read_point_cloud(pcd_path)
    if pcd.is_empty():
        return None
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return np.asarray(pcd.points)
