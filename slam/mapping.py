import os
import numpy as np
import open3d as o3d
from slam.pre_process import preprocess
from slam.registration import icp
from slam.loop_closure import LoopClosureDetector
from slam.graph_optimization import optimize_pose_graph



def build_map(pcd_folder, voxel_size=0.3):
    files = sorted([os.path.join(pcd_folder, f) for f in os.listdir(pcd_folder) if f.endswith(".pcd")])
    if not files: return None
    
    print(f"Lendo {len(files)} arquivos...")
    scans = [preprocess(f, voxel_size) for f in files]
    
    current_pose = np.eye(4)
    map_points = [scans[0]]
    poses = [current_pose.copy()]  # guarde todas as poses
    detector = LoopClosureDetector(threshold=0.11, min_gap=20)
    detector.add_scan(scans[0])  # adiciona o primeiro frame
    loop_closures = []           # lista de pares (i, j) detectados

    for i in range(1, len(scans)):
        print(f"Registrando frame {i-1} → {i}...")
        # Alinha o novo scan (source) ao anterior (target)
        R, t, _ = icp(scans[i], scans[i-1])
        T_rel = np.eye(4)
        T_rel[:3, :3] = R
        T_rel[:3, 3] = t
        
        current_pose = T_rel @ current_pose
        poses.append(current_pose.copy())
        
        # Transforma pontos para o mundo global
        scan_global = (np.dot(current_pose[:3, :3], scans[i].T)).T + current_pose[:3, 3]
        map_points.append(scan_global)
        found, loop_idx = detector.add_scan(scans[i])
        if found:
            loop_closures.append((i, loop_idx))
    
    if loop_closures:
        print(f"\nOtimizando grafo com {len(loop_closures)} loops")
        poses = optimize_pose_graph(poses, loop_closures, scans)
    else:
        print("\nNenhum loop detectado")
    
    map_points = []
    for i, scan in enumerate (scans):
        R_g = poses[i][:3, :3]
        t_g = poses[i][:3, 3]
        map_points.append((R_g @ scan.T).T + t_g)
        
    map_full = np.vstack(map_points)
    pcd_map = o3d.geometry.PointCloud()
    pcd_map.points = o3d.utility.Vector3dVector(map_full)
    o3d.io.write_point_cloud("map_final.pcd", pcd_map)
    print(f"Mapa salvo: {len(map_full)} pontos")
    return pcd_map, poses, loop_closures