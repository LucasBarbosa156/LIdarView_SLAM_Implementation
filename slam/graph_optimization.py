import g2o
import numpy as np
from slam.registration import icp

def optimize_pose_graph(poses, loop_closures, scans, n_iterations=20):
    #Realiza a otimização de grafo de poses para corrigir o drift global.

    optimizer = g2o.SparseOptimizer()
    solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
    algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
    optimizer.set_algorithm(algorithm)

    for i, pose in enumerate(poses):
        v = g2o.VertexSE3()
        v.set_id(i)
        v.set_estimate(g2o.Isometry3d(pose))
        v.set_fixed(i == 0)
        optimizer.add_vertex(v)

    info_odom = np.eye(6)
    info_loop = np.eye(6) * 10  # Peso maior para loops
    # Conecta frames adjacentes (0-1, 1-2, 2-3...)
    for i in range(len(poses) - 1):
        j = i + 1
        
        # T_rel = inv(P_i) @ P_{i+1}
        # Isso extrai a transformação relativa que o ICP calculou originalmente
        T_rel = np.linalg.inv(poses[i]) @ poses[j]
        
        e = g2o.EdgeSE3()
        e.set_vertex(0, optimizer.vertex(i))
        e.set_vertex(1, optimizer.vertex(j))
        e.set_measurement(g2o.Isometry3d(T_rel))
        e.set_information(info_odom)
        optimizer.add_edge(e)

#DICIONAR ARESTAS DE LOOP CLOSURE

    for i, j in loop_closures:
        # i = frame antigo, j = frame atual
        print(f"Refinando loop entre {i} e {j} com ICP...")
        
        # Rodar ICP extra entre os dois scans para garantir o alinhamento
        # scans[j] é o source, scans[i] é o target
        R, t, _ = icp(scans[j], scans[i])
        
        T_rel_loop = np.eye(4)
        T_rel_loop[:3, :3] = R
        T_rel_loop[:3, 3] = t
        
        e = g2o.EdgeSE3()
        e.set_vertex(0, optimizer.vertex(i))
        e.set_vertex(1, optimizer.vertex(j))
        e.set_measurement(g2o.Isometry3d(T_rel_loop))
        e.set_information(info_loop)
        optimizer.add_edge(e)

   
    print("Iniciando otimização do grafo...")
    optimizer.initialize_optimization()
    optimizer.optimize(n_iterations)

    
    optimized_poses = []
    for i in range(len(poses)):
        # Recupera a matriz 4x4 corrigida do vértice i
        mat = optimizer.vertex(i).estimate().matrix()
        optimized_poses.append(mat)

    return optimized_poses