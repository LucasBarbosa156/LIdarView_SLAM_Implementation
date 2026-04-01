from slam.mapping import build_map
import open3d as o3d

resultado, poses, loop_closures = build_map("./data_raw_lidar/", voxel_size=0.3)

if resultado:
    o3d.visualization.draw_geometries([resultado])
