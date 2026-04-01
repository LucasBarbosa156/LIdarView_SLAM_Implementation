import numpy as np


def compute_centroids(P_matched, Q_matched):
    p_bar = np.mean(P_matched, axis=0)
    q_bar = np.mean(Q_matched, axis=0)
    P_centered = P_matched - p_bar
    Q_centered = Q_matched - q_bar
    return p_bar, q_bar, P_centered, Q_centered


def compute_rotation(P_centered, Q_centered):
    W = np.dot(P_centered.T, Q_centered)
    U, S, Vt = np.linalg.svd(W)
    V = Vt.T
    R = np.dot(V, U.T)
    if np.linalg.det(R) < 0:
        V[:, 2] *= -1
        R = np.dot(V, U.T)
    return R


def estimate_transform(P_matched, Q_matched):
    p_bar, q_bar, P_centered, Q_centered = compute_centroids(
        P_matched, Q_matched)
    R = compute_rotation(P_centered, Q_centered)
    t = q_bar - np.dot(R, p_bar)
    return R, t


def find_correspondences(P, Q):
    dist_sq = np.sum(P**2, axis=1)[:, np.newaxis] + \
        np.sum(Q**2, axis=1) - 2 * np.dot(P, Q.T)
    indices = np.argmin(dist_sq, axis=1)
    return P, Q[indices]


def icp(P, Q, max_iterations=50, tolerance=1e-5):
    P_current = P.copy()
    R_total = np.eye(3)
    t_total = np.zeros(3)
    errors = []

    for i in range(max_iterations):
        P_matched, Q_matched = find_correspondences(P_current, Q)
        R, t = estimate_transform(P_matched, Q_matched)

        P_current = np.dot(P_current, R.T) + t
        R_total = np.dot(R, R_total)
        t_total = np.dot(R, t_total) + t

        mean_error = np.mean(np.linalg.norm(P_matched - Q_matched, axis=1))
        errors.append(mean_error)

        if len(errors) > 1 and abs(errors[-2] - errors[-1]) < tolerance:
            print(
                f"  Convergiu na iteração {i}, erro final: {errors[-1]:.4f}m")
            break
        else:
            if i == max_iterations - 1:
                print(
                    f"  Não convergiu após {max_iterations} iterações. Erro final: {errors[-1]:.4f}m")

    return R_total, t_total, errors
