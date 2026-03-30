import numpy as np
import helper as hlp
import skimage.io as io
import submission as sub
import matplotlib.pyplot as plt
import cv2

# 1. Load the two temple images and the points from data/some_corresp.npz
img1 = cv2.imread('im1.png')
img2 = cv2.imread('im2.png')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
data = np.load('some_corresp.npz')
print(data.files)
pts1 = data['pts1']
pts2 = data['pts2']

# 2. Run eight_point to compute F
h, w, _ = img1.shape
M = max(h, w)
F = sub.eight_point(pts1, pts2, M)
#print(F)
#hlp.displayEpipolarF(img1,img2,F)

# 3. Load points in image 1 from data/temple_coords.npz
plt.imshow(img1)
plt.scatter(pts1[:, 0], pts1[:, 1], c='r', s=10)
plt.show(block=False)
plt.pause(2)
plt.close()



# 4. Run epipolar_correspondences to get points in image 2
pts2=sub.epipolar_correspondences(img1,img2,F,pts1)
#print(pts2.shape)
plt.imshow(img2)
plt.scatter(pts2[:, 0], pts2[:, 1], c='r', s=10)
plt.show(block=False)
plt.pause(2)
plt.close()
#hlp.epipolarMatchGUI(img1,img2,F)
data2 = np.load('intrinsics.npz')
print(data2.files)
K1 = data2['K1']
K2 = data2['K2']
E=sub.essential_matrix(F,K1,K2)
print(E)


# 5. Compute the camera projection matrix P1
P1 = np.hstack((np.eye(3), np.zeros((3, 1))))

# 6. Use camera2 to get 4 camera projection matrices P2
P2s=hlp.camera2(E)


# 7. Run triangulate using the projection matrices



# 8. Figure out the correct P2


def find_correct_P2(P1, P2s, pts1, pts2):
    best_P2 = None
    best_pts3d = None
    max_positive_depth = -1

    for P2 in P2s:
        pts3d=sub.triangulate(P1,pts1,P2,pts2)
        pts3d_h = np.hstack((pts3d, np.ones((pts3d.shape[0], 1))))

        depth1 = pts3d[:, 2]
        depth2 = (P2 @ pts3d_h.T)[2]

        num_positive = np.sum((depth1 > 0) & (depth2 > 0))

        if num_positive > max_positive_depth:
            max_positive_depth = num_positive
            best_P2 = P2

    return best_P2
# 9. Scatter plot the correct 3D points

import matplotlib.pyplot as plt
import numpy as np

def plot_3d_points(pts3d):
    # --- Normalize the point cloud ---
    pts = pts3d.copy()

    # 1. Center (move centroid to origin)
    centroid = np.mean(pts, axis=0)
    pts -= centroid

    # 2. Scale (bring values to reasonable range)
    max_dist = np.max(np.linalg.norm(pts, axis=1))
    if max_dist > 0:
        pts /= max_dist   # now roughly in [-1, 1]

    # --- Plot ---
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    ax.scatter(x, y, z, s=5)

    # Equal aspect ratio
    ax.set_box_aspect([1, 1, 1])

    # Optional: clean axis limits
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    # Optional: nicer ticks (now 0.2 makes sense)
    ticks = np.arange(-1, 1.2, 0.2)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)

    ax.set_title("3D Reconstruction")

    plt.show()
# 10. Save the computed extrinsic parameters (R1,R2,t1,t2) to data/extrinsics.npz
def save_extrinsics(P1, P2, filename='extrinsics.npz'):
    R1 = P1[:, :3]
    t1 = P1[:, 3]

    R2 = P2[:, :3]
    t2 = P2[:, 3]

    np.savez(filename, R1=R1, t1=t1, R2=R2, t2=t2)



P2 = find_correct_P2(P1, P2s, pts1, pts2)

print("Correct P2:\n", P2)
pts3d = sub.triangulate(P1, pts1, P2, pts2)
plot_3d_points(pts3d)
save_extrinsics(P1, P2)

print("Extrinsics saved to data/extrinsics.npz")