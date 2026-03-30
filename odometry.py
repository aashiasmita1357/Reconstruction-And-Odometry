import cv2
import numpy as np
from submission import (
    eight_point,
    triangulate,
    estimate_pose,
    estimate_params
)
from odometry_visualizer import TrajectoryVisualizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

K = np.array([[517.3, 0, 318.6],
              [0, 516.5, 255.3],
              [0, 0, 1]])

cap = cv2.VideoCapture("dataset.mp4")
if not cap.isOpened():
    raise ValueError("Cannot open video file")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter('output_trajectory.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps, (frame_width, frame_height))
ret, frame_prev = cap.read()
if not ret:
    raise ValueError("Cannot read first frame")
frame_prev_gray = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)

feature_params = dict(maxCorners=300, qualityLevel=0.01, minDistance=10)
pts_prev = cv2.goodFeaturesToTrack(frame_prev_gray, mask=None, **feature_params)
pts_prev = pts_prev.reshape(-1, 2)

R_f = np.eye(3)
t_f = np.zeros((3, 1))

vis = TrajectoryVisualizer()
vis.add_pose(t_f.flatten())
vis.visualize()

frame_idx = 1
while True:
    ret, frame_curr = cap.read()
    if not ret:
        break

    frame_curr_gray = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2GRAY)
    pts_curr, status, _ = cv2.calcOpticalFlowPyrLK(
        frame_prev_gray, frame_curr_gray, pts_prev.astype(np.float32), None)

    mask = (status.flatten() == 1)
    good_prev = pts_prev[mask]
    good_curr = pts_curr[mask]

    E, mask_e = cv2.findEssentialMat(good_curr, good_prev, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask_pose = cv2.recoverPose(E, good_curr, good_prev, K, mask=mask_e)
    scale = 1.0 
    t_f = t_f + scale * (R_f @ t)
    R_f = R @ R_f
    vis.add_pose(t_f.flatten())
    vis.visualize()

    vis.fig.canvas.draw()
    w_vis, h_vis = vis.fig.canvas.get_width_height()
    buf = np.frombuffer(vis.fig.canvas.tostring_argb(), dtype=np.uint8)
    buf = buf.reshape(h_vis, w_vis, 4)
 
    img_plot = buf[:, :, [3, 2, 1]] 
    img_plot = cv2.resize(img_plot, (frame_width, frame_height))
    out.write(img_plot)

    if len(good_curr) < 150:
        new_pts = cv2.goodFeaturesToTrack(frame_curr_gray, mask=None, **feature_params)
        pts_prev = new_pts.reshape(-1, 2)
    else:
        pts_prev = good_curr
        
    frame_prev_gray = frame_curr_gray.copy()
    frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()
print("Trajectory video saved as output_trajectory.mp4")