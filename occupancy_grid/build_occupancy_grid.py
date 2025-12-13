import numpy as np
import cv2
import os
from bresenham import bresenham
import subprocess

# ----------------------------
# PARAMETERS
# ----------------------------
depth_input_folder = "images"
depth_output_folder = "depth_frames"
map_width, map_height = 500, 500
resolution = 0.05
origin = (map_width//2, map_height//2)
max_depth = 5.0
fx, fy, cx, cy = 525.0, 525.0, 319.5, 239.5 # Camera intrinsics: should be adjustsed based on your camera

# ----------------------------
# 1. RUN DEPTH ANYTHING (v2)
# ----------------------------
# Using subprocess to call the script with numpy output
subprocess.run([
    "python", "metric_depth/run.py",
    "--input_folder", depth_input_folder,
    "--save", "numpy"
])

# ----------------------------
# 2. RUN PYSLAM MAIN
# ----------------------------
# Assuming main_slam.py has a function that returns a dict of {timestamp: (x, y, theta)}
# You may need to modify main_slam.py to expose a function like get_poses()
from main_slam import get_poses  # you will need to add this function in main_slam.py or main_slam_copy.py

poses = get_poses(depth_input_folder)  # returns {timestamp: (x, y, theta)}

# ----------------------------
# 3. INITIALIZE OCCUPANCY GRID
# ----------------------------
grid = -1 * np.ones((map_height, map_width), dtype=np.int8)

# ----------------------------
# 4. HELPER FUNCTIONS
# ----------------------------
def world_to_grid(x, y):
    gx = int(x / resolution + origin[0])
    gy = int(y / resolution + origin[1])
    return gx, gy

def depth_to_points(depth):
    H, W = depth.shape
    i, j = np.indices((H, W))
    z = depth
    x = (j - cx) * z / fx
    y = (i - cy) * z / fy
    mask = (z > 0) & (z < max_depth)
    return np.stack((x[mask], z[mask]), axis=-1)

def transform_to_world(points, pose):
    x_r, y_r, theta = pose
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s],
                  [s,  c]])
    return points @ R.T + np.array([x_r, y_r])

def update_grid(grid, robot_pose, points):
    gx0, gy0 = world_to_grid(robot_pose[0], robot_pose[1])
    for px, py in points:
        gx, gy = world_to_grid(px, py)
        for x_cell, y_cell in bresenham(gx0, gy0, gx, gy):
            if 0 <= x_cell < map_width and 0 <= y_cell < map_height:
                grid[y_cell, x_cell] = 0
        if 0 <= gx < map_width and 0 <= gy < map_height:
            grid[gy, gx] = 1

# ----------------------------
# 5. LIVE VISUALIZATION
# ----------------------------
cv2.namedWindow("Occupancy Grid", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Occupancy Grid", 600, 600)

depth_files = sorted(os.listdir(depth_output_folder))
for file_name in depth_files:
    if not file_name.endswith(".npy"):
        continue
    timestamp = float(os.path.splitext(file_name)[0])
    if timestamp not in poses:
        continue

    depth_path = os.path.join(depth_output_folder, file_name)
    depth = np.load(depth_path)
    points = depth_to_points(depth)
    points_world = transform_to_world(points, poses[timestamp])
    update_grid(grid, poses[timestamp], points_world)

    # display
    display_grid = np.zeros_like(grid, dtype=np.uint8)
    display_grid[grid == -1] = 127
    display_grid[grid == 0] = 255
    display_grid[grid == 1] = 0
    cv2.imshow("Occupancy Grid", display_grid)
    key = cv2.waitKey(1)
    if key == 27:  # ESC to quit
        break

cv2.waitKey(0)
cv2.destroyAllWindows()
np.save("occupancy_grid.npy", grid)
