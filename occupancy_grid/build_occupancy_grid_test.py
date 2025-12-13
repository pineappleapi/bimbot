#!/usr/bin/env -S python3 -O
# -*- coding: utf-8 -*-
"""
Square room exploration test
----------------------------
Simulates a robot exploring a square room:
- Generates a pose file (poses.txt) with timestamps and (x,y,theta) along a square/path.
- Generates synthetic depth frames (depth_frames/*.npy) by raycasting to square room walls
  from each pose (using a simple 2D pinhole model).
- Builds a 2D occupancy grid and visualizes it with Matplotlib (grid lines, axis ticks in meters,
  robot pose arrow and path overlay).
- Saves occupancy_grid.npy and occupancy_grid.png.

Run examples:
    python square_explore_test.py --room-side 10.0 --trajectory square --num-poses 120
    python square_explore_test.py --room-side 8.0 --trajectory lawnmower --num-poses 180 --headless 1

Requirements:
    pip install numpy matplotlib

Notes:
- Axes convention: points are (x,z) in robot frame: x=lateral (left/right), z=forward.
- Pose is (x_r, y_r, theta) in world meters/radians. R rotates local points to world.
- Depth image uses HxW = 480x640 and intrinsics FX,FY,CX,CY.
"""

import os
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow

# ----------------------------
# Config
# ----------------------------
MAP_WIDTH, MAP_HEIGHT = 500, 500
RESOLUTION = 0.05  # m per cell
ORIGIN = (MAP_WIDTH // 2, MAP_HEIGHT // 2)
MAX_DEPTH = 10.0

FX, FY, CX, CY = 525.0, 525.0, 319.5, 239.5
IMG_H, IMG_W = 480, 640
DEPTH_DIR = "depth_frames"
POSE_FILE = "poses.txt"

# ----------------------------
# Bresenham
# ----------------------------

def bresenham(x0, y0, x1, y1):
    dx = abs(x1 - x0); dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            yield x, y
            err -= dy
            if err < 0:
                y += sy; err += dx
            x += sx
        yield x, y
    else:
        err = dy / 2.0
        while y != y1:
            yield x, y
            err -= dx
            if err < 0:
                x += sx; err += dy
            y += sy
        yield x, y

# ----------------------------
# Grid helpers
# ----------------------------

def init_grid(w, h):
    return -1 * np.ones((h, w), dtype=np.int8)

def world_to_grid(x, y, resolution=RESOLUTION, origin=ORIGIN):
    return int(x / resolution + origin[0]), int(y / resolution + origin[1])

def update_grid(grid, robot_pose, points_world):
    gx0, gy0 = world_to_grid(robot_pose[0], robot_pose[1])
    h, w = grid.shape
    for px, py in points_world:
        gx, gy = world_to_grid(px, py)
        for x_cell, y_cell in bresenham(gx0, gy0, gx, gy):
            if 0 <= x_cell < w and 0 <= y_cell < h:
                grid[y_cell, x_cell] = 0
        if 0 <= gx < w and 0 <= gy < h:
            grid[gy, gx] = 1

# ----------------------------
# Matplotlib viz
# ----------------------------

def meters_per_tick(resolution, w, h):
    meters_across = min(w, h) * resolution
    approx = meters_across / 20.0
    for nice in [0.1, 0.2, 0.5, 1.0, 2.0]:
        if approx <= nice: return nice
    return 2.0

def render_display_grid(grid):
    display = np.zeros_like(grid, dtype=np.uint8)
    display[grid == -1] = 127
    display[grid == 0] = 255
    display[grid == 1] = 0
    return display

def imshow_with_grid(ax, grid, resolution=RESOLUTION, origin=ORIGIN, show_meters=True):
    h, w = grid.shape
    im = ax.imshow(render_display_grid(grid), cmap='gray', origin='lower', interpolation='nearest')
    ax.set_title('Occupancy Grid')
    tick_m = meters_per_tick(resolution, w, h)
    tick_cells = max(1, int(round(tick_m / resolution)))
    xt = np.arange(0, w, tick_cells)
    yt = np.arange(0, h, tick_cells)
    ax.set_xticks(xt); ax.set_yticks(yt)
    if show_meters:
        ax.set_xticklabels([f'{(x-origin[0])*resolution:.1f}' for x in xt])
        ax.set_yticklabels([f'{(y-origin[1])*resolution:.1f}' for y in yt])
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    else:
        ax.set_xlabel('Grid X (cells)'); ax.set_ylabel('Grid Y (cells)')
    ax.grid(True, color='lightgray', linewidth=0.5)
    ax.scatter([origin[0]], [origin[1]], c='red', s=30, label='Origin')
    ax.legend(loc='upper right')
    return im

def draw_robot_pose(ax, pose, resolution=RESOLUTION, origin=ORIGIN, arrow_len_m=0.5, color='tab:blue'):
    x_r, y_r, theta = pose
    def w2g(x, y): return int(x/resolution + origin[0]), int(y/resolution + origin[1])
    gx, gy = w2g(x_r, y_r)
    xt = x_r + arrow_len_m * math.cos(theta)
    yt = y_r + arrow_len_m * math.sin(theta)
    gxt, gyt = w2g(xt, yt)
    arrow = FancyArrow(gx, gy, gxt-gx, gyt-gy, width=0.0, head_width=5.0, head_length=7.0,
                       length_includes_head=True, color=color)
    ax.add_patch(arrow)
    ax.plot(gx, gy, 'o', color=color, markersize=4)

# ----------------------------
# Pose generation (square perimeter or lawnmower)
# ----------------------------

def generate_poses(room_side_m=10.0, num_poses=120, trajectory='square', base_ts=1_700_000_000.0):
    """Return list of (ts, x, y, theta). Theta = heading along path."""
    L = room_side_m
    half = L/2.0
    poses = []
    if trajectory == 'square':
        # Perimeter walk: 4 edges evenly
        edge = num_poses // 4
        # Start at (-half+0.5, -half+0.5), go CCW
        x, y = -half+0.5, -half+0.5
        # Right edge (increase x)
        for k in range(edge):
            t = base_ts + 0.1*len(poses)
            s = k/edge
            px = -half+0.5 + s*(L-1.0)
            py = -half+0.5
            theta = 0.0  # facing +x
            poses.append((t, px, py, theta))
        # Top (increase y)
        for k in range(edge):
            t = base_ts + 0.1*len(poses)
            s = k/edge
            px = half-0.5
            py = -half+0.5 + s*(L-1.0)
            theta = math.pi/2  # +y
            poses.append((t, px, py, theta))
        # Left (decrease x)
        for k in range(edge):
            t = base_ts + 0.1*len(poses)
            s = k/edge
            px = half-0.5 - s*(L-1.0)
            py = half-0.5
            theta = math.pi  # -x
            poses.append((t, px, py, theta))
        # Bottom (decrease y)
        remaining = num_poses - 3*edge
        for k in range(remaining):
            t = base_ts + 0.1*len(poses)
            s = k/remaining if remaining>0 else 0.0
            px = -half+0.5
            py = half-0.5 - s*(L-1.0)
            theta = -math.pi/2  # -y
            poses.append((t, px, py, theta))
    else:
        # lawnmower rows across X, sweeping Y
        rows = max(3, num_poses//20)
        cols_per_row = max(10, num_poses//rows)
        xs = np.linspace(-half+0.5, half-0.5, cols_per_row)
        ys = np.linspace(-half+0.5, half-0.5, rows)
        flip = False
        for r, y in enumerate(ys):
            x_line = xs if not flip else xs[::-1]
            for c, x in enumerate(x_line):
                t = base_ts + 0.1*len(poses)
                # heading along line
                if c < len(x_line)-1:
                    nx = x_line[c+1]; ny = y
                    theta = math.atan2(ny - y, nx - x)
                else:
                    theta = 0.0
                poses.append((t, float(x), float(y), float(theta)))
            flip = not flip
    return poses

# ----------------------------
# Depth generation by raycasting to square walls
# ----------------------------

def ray_depth_to_square_walls(pose, room_side_m, point_stride=2):
    """
    Generate a synthetic depth image HxW for a square room with side 'room_side_m'.
    For each column j, compute the ray angle alpha = arctan((j-cx)/fx) in robot frame.
    Ray direction v_r = [sin(alpha), cos(alpha)] (lateral, forward). Transform to world via R.
    Intersect with square walls (x=±L/2, y=±L/2), choose nearest positive t.
    Depth per pixel is z_forward = t * cos(alpha).
    We fill the whole row with the same depth to emulate vertical consistency.
    """
    L = room_side_m
    half = L/2.0
    x_r, y_r, theta = pose
    c, s = math.cos(theta), math.sin(theta)

    depth = np.zeros((IMG_H, IMG_W), dtype=np.float32)

    # Precompute a subset of columns for stride, then broadcast across rows
    cols = np.arange(0, IMG_W, point_stride)
    alphas = np.arctan2((cols - CX), FX)  # horizontal angles

    # For each alpha, compute nearest intersection
    z_values = np.full_like(cols, fill_value=MAX_DEPTH, dtype=np.float32)
    for idx, alpha in enumerate(alphas):
        # Robot-frame unit ray
        v_rx = math.sin(alpha)
        v_ry = math.cos(alpha)
        # World-frame ray via rotation R
        v_wx = c*v_rx - s*v_ry
        v_wy = s*v_rx + c*v_ry
        # Avoid zero directions
        eps = 1e-6
        if abs(v_wx) < eps and abs(v_wy) < eps:
            z_values[idx] = MAX_DEPTH
            continue
        candidates = []
        # Intersect x=+half and x=-half
        if abs(v_wx) >= eps:
            t1 = ( half - x_r) / v_wx
            y1 = y_r + t1*v_wy
            if t1 > 0 and -half - 1e-6 <= y1 <= half + 1e-6:
                candidates.append(t1)
            t2 = (-half - x_r) / v_wx
            y2 = y_r + t2*v_wy
            if t2 > 0 and -half - 1e-6 <= y2 <= half + 1e-6:
                candidates.append(t2)
        # Intersect y=+half and y=-half
        if abs(v_wy) >= eps:
            t3 = ( half - y_r) / v_wy
            x3 = x_r + t3*v_wx
            if t3 > 0 and -half - 1e-6 <= x3 <= half + 1e-6:
                candidates.append(t3)
            t4 = (-half - y_r) / v_wy
            x4 = x_r + t4*v_wx
            if t4 > 0 and -half - 1e-6 <= x4 <= half + 1e-6:
                candidates.append(t4)
        if candidates:
            t = min(candidates)
            z_forward = t * v_ry  # forward component in robot frame
            # Clamp
            z_values[idx] = np.float32(max(0.1, min(z_forward, MAX_DEPTH)))
        else:
            z_values[idx] = np.float32(MAX_DEPTH)

    # Broadcast z_values to full image rows at stride positions
    for r in range(IMG_H):
        depth[r, cols] = z_values
        # Fill non-stride columns by nearest-neighbor
        for k in range(len(cols)-1):
            start = cols[k]; end = cols[k+1]
            depth[r, start:end] = z_values[k]
        # last segment
        if len(cols) > 0:
            depth[r, cols[-1]:IMG_W] = z_values[-1]
    return depth

# ----------------------------
# Load/save poses
# ----------------------------

def save_poses_txt(poses, path=POSE_FILE):
    with open(path, 'w', encoding='utf-8') as f:
        f.write("timestamp,x,y,z,theta\n")
        for (ts, x, y, theta) in poses:
            f.write(f"{ts:.6f},{x:.6f},{y:.6f},0.000000,{theta:.6f}\n")


def load_poses_from_txt(path=POSE_FILE):
    data = np.genfromtxt(path, delimiter=',', skip_header=1)
    poses = {}
    for row in data:
        ts, x, y, z, theta = row
        poses[float(ts)] = (float(x), float(y), float(theta))
    return poses

# ----------------------------
# Transform & projection helpers
# ----------------------------

def depth_to_points(depth, max_depth=MAX_DEPTH, point_stride=2):
    h, w = depth.shape
    i, j = np.indices((h, w))
    z = depth
    x = (j - CX) * z / FX
    mask = (z > 0) & (z < max_depth) & ((i % point_stride == 0) & (j % point_stride == 0))
    return np.stack((x[mask], z[mask]), axis=-1)


def transform_to_world(points, pose):
    x_r, y_r, theta = pose
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    return points @ R.T + np.array([x_r, y_r], dtype=np.float32)

# ----------------------------
# Main
# ----------------------------

def main():
    global RESOLUTION, ORIGIN
    ap = argparse.ArgumentParser(description="Square room exploration test")
    ap.add_argument('--room-side', type=float, default=10.0, help='Square room side length (m)')
    ap.add_argument('--trajectory', type=str, default='square', choices=['square','lawnmower'], help='Path type')
    ap.add_argument('--num-poses', type=int, default=160, help='Number of poses')
    ap.add_argument('--point-stride', type=int, default=3, help='Depth sampling stride')
    ap.add_argument('--map-width', type=int, default=MAP_WIDTH)
    ap.add_argument('--map-height', type=int, default=MAP_HEIGHT)
    ap.add_argument('--resolution', type=float, default=RESOLUTION)
    ap.add_argument('--headless', type=int, default=0)
    args = ap.parse_args()

    # Generate poses
    print(f"[*] Generating poses for a {args.room_side} m square room...")
    poses_list = generate_poses(room_side_m=args.room_side, num_poses=args.num_poses, trajectory=args.trajectory)
    save_poses_txt(poses_list, POSE_FILE)
    poses = load_poses_from_txt(POSE_FILE)

    # Generate depth frames per pose
    os.makedirs(DEPTH_DIR, exist_ok=True)
    print(f"[*] Generating synthetic depth frames in {DEPTH_DIR}/ ...")
    for (ts, x, y, theta) in poses_list:
        depth = ray_depth_to_square_walls((x, y, theta), args.room_side, point_stride=max(1, args.point_stride))
        np.save(os.path.join(DEPTH_DIR, f"{ts:.6f}.npy"), depth.astype(np.float32))

    # Initialize grid
    RESOLUTION = args.resolution
    grid = init_grid(args.map_width, args.map_height)
    ORIGIN = (args.map_width // 2, args.map_height // 2)

    # Matplotlib figure
    fig, ax = plt.subplots(figsize=(8,8))
    im = imshow_with_grid(ax, grid, resolution=RESOLUTION, origin=ORIGIN, show_meters=True)
    plt.ion(); fig.canvas.draw(); fig.canvas.flush_events()

    path_xy = []
    # Iterate frames in timestamp order
    depth_files = [n for n in sorted(os.listdir(DEPTH_DIR)) if n.endswith('.npy')]
    print(f"[*] Mapping {len(depth_files)} frames...")
    for name in depth_files:
        ts = float(os.path.splitext(name)[0])
        pose = poses.get(ts, None)
        if pose is None:
            print(f"[!] Missing pose for timestamp {ts:.6f}; skipping.")
            continue
        depth = np.load(os.path.join(DEPTH_DIR, name))
        points = depth_to_points(depth, max_depth=MAX_DEPTH, point_stride=args.point_stride)
        points_world = transform_to_world(points, pose)
        update_grid(grid, pose, points_world)

        # Update viz
        
        im.set_data(render_display_grid(grid))

        # Remove previous overlays (origin scatter, pose arrow, prior path line)
        for coll in list(ax.collections):
            try:
                coll.remove()
            except Exception:
                pass
        for p in list(ax.patches):
            try:
                p.remove()
            except Exception:
                pass
        for ln in list(ax.lines):
            try:
                ln.remove()
            except Exception:
                pass

        # Re-draw origin marker if you want it persistent
        ax.scatter([ORIGIN[0]], [ORIGIN[1]], c='red', s=30, label='Origin')

        # Pose arrow and path
        draw_robot_pose(ax, pose, resolution=RESOLUTION, origin=ORIGIN, arrow_len_m=0.6)
        path_xy.append((pose[0], pose[1]))
        if len(path_xy) > 1:
            gxgy = np.array([[int(x/RESOLUTION + ORIGIN[0]), int(y/RESOLUTION + ORIGIN[1])] for (x, y) in path_xy])
            ax.plot(gxgy[:,0], gxgy[:,1], color='tab:orange', linewidth=1.0)

        fig.canvas.draw(); fig.canvas.flush_events()
        plt.pause(0.01)


    # Save outputs
    np.save("occupancy_grid.npy", grid)
    fig.savefig("occupancy_grid.png", dpi=180, bbox_inches='tight')
    print("[✓] Saved occupancy_grid.npy and occupancy_grid.png")
    plt.ioff()
    if not args.headless:
        print("[✓] Close the figure to finish.")
        plt.show()

if __name__ == '__main__':
    main()
