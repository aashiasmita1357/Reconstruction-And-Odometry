"""
Homework 5
Submission Functions
"""

# import packages here
import cv2
import numpy as np
from scipy.linalg import rq

"""
Q3.1.1 Eight Point Algorithm
       [I] pts1, points in image 1 (Nx2 matrix)
           pts2, points in image 2 (Nx2 matrix)
           M, scalar value computed as max(H1,W1)
       [O] F, the fundamental matrix (3x3 matrix)
"""
def eight_point(pts1, pts2, M):
    # replace pass by your implementation
    T = np.array([
    [1/M, 0,   0],
    [0,   1/M, 0],
    [0,   0,   1]
    ])
    ones = np.ones((pts1.shape[0], 1))
    pts1_h = np.hstack([pts1, ones])
    pts2_h = np.hstack([pts2, ones])
    pts1_norm = (T @ pts1_h.T).T
    pts2_norm = (T @ pts2_h.T).T

    A = []
    for i in range(pts1.shape[0]):
        x1, y1 = pts1_norm[i, 0], pts1_norm[i, 1]
        x2, y2 = pts2_norm[i, 0], pts2_norm[i, 1]

        A.append([
            x2*x1, x2*y1, x2,
            y2*x1, y2*y1, y2,
            x1,    y1,    1
        ])

    A = np.array(A)

    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)

    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0
    F = U @ np.diag(S) @ Vt

    F = T.T @ F @ T
    return F / F[-1, -1]


"""
Q3.1.2 Epipolar Correspondences
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           F, fundamental matrix from image 1 to image 2 (3x3 matrix)
           pts1, points in image 1 (Nx2 matrix)
       [O] pts2, points in image 2 (Nx2 matrix)
"""
def epipolar_correspondences(im1, im2, F, pts1):
    # replace pass by your implementation
    img1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    window_size = 5
    half = window_size // 2

    pts2 = []

    for (x1, y1) in pts1:
        x1, y1 = int(x1), int(y1)

        p1 = np.array([x1, y1, 1])
        l = F @ p1
        a, b, c = l

        best_error = float('inf')
        best_point = (0, 0)

        patch1 = img1_gray[y1-half:y1+half+1,
                           x1-half:x1+half+1]

        for x2 in range(half, im2.shape[1]-half):
            y2 = int(-(a*x2 + c)/b)

            if y2 < half or y2 >= im2.shape[0]-half:
                continue

            patch2 = img2_gray[y2-half:y2+half+1,
                               x2-half:x2+half+1]

            error = np.sum((patch1 - patch2)**2)

            if error < best_error:
                best_error = error
                best_point = (x2, y2)

        pts2.append(best_point)

    return np.array(pts2) 


"""
Q3.1.3 Essential Matrix
       [I] F, the fundamental matrix (3x3 matrix)
           K1, camera matrix 1 (3x3 matrix)
           K2, camera matrix 2 (3x3 matrix)
       [O] E, the essential matrix (3x3 matrix)
"""
def essential_matrix(F, K1, K2):
    # replace pass by your implementation
    E = K2.T @ F @ K1

    U, S, Vt = np.linalg.svd(E)
    S = [1, 1, 0]
    E = U @ np.diag(S) @ Vt

    return E


"""
Q3.1.4 Triangulation
       [I] P1, camera projection matrix 1 (3x4 matrix)
           pts1, points in image 1 (Nx2 matrix)
           P2, camera projection matrix 2 (3x4 matrix)
           pts2, points in image 2 (Nx2 matrix)
       [O] pts3d, 3D points in space (Nx3 matrix)
"""
def triangulate(P1, pts1, P2, pts2):
    # replace pass by your implementation
    N = pts1.shape[0]
    pts3d = []
    print(pts1.shape, pts2.shape)
    print(type(pts1[0]), type(pts2[0]))
    print("P1 shape:", P1.shape)
    print("P2 shape:", P2.shape)

    for i in range(N):
        x1, y1 = float(pts1[i][0]), float(pts1[i][1])
        x2, y2 = float(pts2[i][0]), float(pts2[i][1])

        A = np.array([
            x1 * P1[2] - P1[0],
            y1 * P1[2] - P1[1],
            x2 * P2[2] - P2[0],
            y2 * P2[2] - P2[1]
        ])

        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X = X / X[3]  # normalize

        pts3d.append(X[:3])

    return np.array(pts3d)



"""
Q3.2.1 Image Rectification
       [I] K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] M1 M2, rectification matrices (3x3 matrix)
           K1p K2p, rectified camera matrices (3x3 matrix)
           R1p R2p, rectified rotation matrices (3x3 matrix)
           t1p t2p, rectified translation vectors (3x1 matrix)
"""
def rectify_pair(K1, K2, R1, R2, t1, t2):
    # replace pass by your implementation
    c1 = -R1.T @ t1
    c2 = -R2.T @ t2

    r1 = (c2 - c1).reshape(3)
    r1 = r1 / np.linalg.norm(r1)

    r2 = np.cross(R1[2], r1)
    r2 = r2 / np.linalg.norm(r2)

    r3 = np.cross(r1, r2)

    R_rect = np.vstack((r1, r2, r3))

    R1p = R_rect
    R2p = R_rect

    K1p = K1
    K2p = K2

    t1p = -R_rect @ c1
    t2p = -R_rect @ c2

    M1 = K1p @ R1p @ np.linalg.inv(K1 @ R1)
    M2 = K2p @ R2p @ np.linalg.inv(K2 @ R2)

    return M1, M2, K1p, K2p, R1p, R2p, t1p, t2p


"""
Q3.2.2 Disparity Map
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           max_disp, scalar maximum disparity value
           win_size, scalar window size value
       [O] dispM, disparity map (H1xW1 matrix)
"""
def get_disparity(im1, im2, max_disp, win_size):
    # replace pass by your implementation
    im1 = im1.astype(float)
    im2 = im2.astype(float)

    H, W = im1.shape
    dispM = np.zeros((H, W))

    half = win_size // 2

    for y in range(half, H-half):
        for x in range(half, W-half):

            best_disp = 0
            min_error = float('inf')

            patch1 = im1[y-half:y+half+1, x-half:x+half+1]

            for d in range(max_disp):
                x2 = x - d
                if x2 - half < 0:
                    continue

                patch2 = im2[y-half:y+half+1, x2-half:x2+half+1]

                error = np.sum((patch1 - patch2)**2)

                if error < min_error:
                    min_error = error
                    best_disp = d

            dispM[y, x] = best_disp

    return dispM


"""
Q3.2.3 Depth Map
       [I] dispM, disparity map (H1xW1 matrix)
           K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] depthM, depth map (H1xW1 matrix)
"""
def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    # replace pass by your implementation
    depthM = np.zeros_like(dispM)

    f = K1[0, 0]

    c1 = -R1.T @ t1
    c2 = -R2.T @ t2
    B = np.linalg.norm(c1 - c2)

    mask = dispM > 0

    depthM[mask] = (f * B) / dispM[mask]

    return depthM


"""
Q3.3.1 Camera Matrix Estimation
       [I] x, 2D points (Nx2 matrix)
           X, 3D points (Nx3 matrix)
       [O] P, camera matrix (3x4 matrix)
"""
def estimate_pose(x, X):
    # replace pass by your implementation
    N = x.shape[0]
    A = []

    for i in range(N):
        Xw, Yw, Zw = X[i]
        u, v = x[i]

        A.append([Xw, Yw, Zw, 1, 0, 0, 0, 0, -u*Xw, -u*Yw, -u*Zw, -u])
        A.append([0, 0, 0, 0, Xw, Yw, Zw, 1, -v*Xw, -v*Yw, -v*Zw, -v])

    A = np.array(A)

    _, _, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3, 4)

    return P


"""
Q3.3.2 Camera Parameter Estimation
       [I] P, camera matrix (3x4 matrix)
       [O] K, camera intrinsics (3x3 matrix)
           R, camera extrinsics rotation (3x3 matrix)
           t, camera extrinsics translation (3x1 matrix)
"""
def estimate_params(P):
    # replace pass by your implementation
    M = P[:, :3]

    K, R = rq(M)

    T = np.diag(np.sign(np.diag(K)))
    K = K @ T
    R = T @ R

    K = K / K[2, 2]

    t = np.linalg.inv(K) @ P[:, 3]

    return K, R, t
