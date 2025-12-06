
# build_occupancy_grid.py
# (see top-of-file comments for usage)

import argparse
import csv
import json
import os
from typing import Dict, Tuple, Optional

import numpy as np
import cv2

# ------------------------- CLI -------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Build 2D occupancy grid from intrinsics, poses, and depth .npy files.')
    p.add_argument('--intrinsics', required=True, help='D:\DepthAnything\Depth-Anything-V2\intrinsics.json')
    p.add_argument('--poses', required=True, help='Path to PySLAM poses CSV.')
    p.add_argument('--depth-dir', required=True, help='Directory containing depth .npy files.')
    p.add_argument('--out-dir', required=True, help='Output directory for map assets.')
    # Grid config
    p.add_argument('--grid-res', type=float, default=0.05, help='Grid resolution in meters per cell (default 0.05).')
    p.add_argument('--grid-width', type=float, default=40.0, help='Grid width in meters (default 40).')
    p.add_argument('--grid-height', type=float, default=40.0, help='Grid height in meters (default 40).')
    p.add_argument('--origin-x', type=float, default=-20.0, help='World X of grid[0,0] in meters (default -20).')
    p.add_argument('--origin-y', type=float, default=-20.0, help='World Y of grid[0,0] in meters (default -20).')
    # Raycasting / updates
    p.add_argument('--stride', type=int, default=4, help='Pixel sampling stride for raycasting (default 4).')
    p.add_argument('--max-depth', type=float, default=20.0, help='Max depth to accept (meters).')
    p.add_argument('--l-free', type=float, default=-0.4, help='Log-odds update for free cells.')
    p.add_argument('--l-occ', type=float, default=+0.85, help='Log-odds update for occupied cells.')
    p.add_argument('--l-min', type=float, default=-4.0, help='Min clamp for log-odds.')
    p.add_argument('--l-max', type=float, default=+4.0, help='Max clamp for log-odds.')
    # Optional height filter
    p.add_argument('--z-min', type=float, default=-np.inf, help='Min world Z to keep (meters).')
    p.add_argument('--z-max', type=float, default=np.inf, help='Max world Z to keep (meters).')
    # Web export
    p.add_argument('--depth-scale', type=float, default=1.0, help='Multiply depths by this scale (default 1.0).')
    p.add_argument('--web-png', action='store_true', help='Also export a web-scaled PNG for dashboards.')
    p.add_argument('--web-width', type=int, default=1024, help='Width of web PNG (default 1024).')
    return p.parse_args()

# ------------------------- Helpers -------------------------

def load_intrinsics(path:str) -> Tuple[float,float,float,float]:
    with open(path,'r') as f:
        j = json.load(f)
    for k in ('fx','fy','cx','cy'):
        if k not in j:
            raise ValueError(f"Missing '{k}' in intrinsics JSON: {path}")
    return float(j['fx']), float(j['fy']), float(j['cx']), float(j['cy'])


def parse_pose_row(row:Dict[str,str]) -> Tuple[int,np.ndarray]:
    '''
    Supports two formats:
      A) frame_idx,T00,T01,...,T33  (16 numeric fields after frame_idx)
      B) frame_idx,T_wc  (JSON string of 4x4)
    Returns: (frame_idx, T_wc 4x4 np.ndarray)
    '''
    frame_idx = int(row.get('frame_idx') or row.get('frame') or row.get('idx') or row.get('id'))
    if 'T_wc' in row and row['T_wc']:
        T = np.array(json.loads(row['T_wc']), dtype=np.float32)
        if T.shape != (4,4):
            raise ValueError('T_wc JSON must be 4x4')
        return frame_idx, T
    # else read 16 numeric columns
    keys = [f'T{i}{j}' for i in range(4) for j in range(4)]
    vals = []
    for k in keys:
        if k not in row:
            raise ValueError('Pose CSV must contain T00..T33 or a T_wc JSON column')
        vals.append(float(row[k]))
    T = np.array(vals, dtype=np.float32).reshape(4,4)
    return frame_idx, T


def load_poses_csv(path:str) -> Dict[int,np.ndarray]:
    poses: Dict[int,np.ndarray] = {}
    with open(path,'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx, T = parse_pose_row(row)
            poses[idx] = T
    return poses


def world_to_grid(x:float, y:float, origin_x:float, origin_y:float, res:float) -> Tuple[int,int]:
    gx = int((x - origin_x)/res)
    gy = int((y - origin_y)/res)
    return gx, gy


def bresenham_line(x0,y0,x1,y1):
    x0,y0,x1,y1 = int(x0),int(y0),int(x1),int(y1)
    pts=[]; dx=abs(x1-x0); dy=-abs(y1-y0)
    sx=1 if x0<x1 else -1
    sy=1 if y0<y1 else -1
    err=dx+dy
    while True:
        pts.append((x0,y0))
        if x0==x1 and y0==y1: break
        e2=2*err
        if e2>=dy:
            err+=dy; x0+=sx
        if e2<=dx:
            err+=dx; y0+=sy
    return pts

# ------------------------- Core -------------------------

def integrate_frame(depth:np.ndarray, K:Tuple[float,float,float,float], T_wc:np.ndarray,
                    grid:np.ndarray, res:float, origin_x:float, origin_y:float,
                    l_free:float, l_occ:float, l_min:float, l_max:float,
                    stride:int, max_depth:float, z_min:float, z_max:float):
    fx,fy,cx,cy = K
    Hd,Wd = depth.shape
    cam_o = T_wc[:3,3]
    for v in range(0,Hd,stride):
        for u in range(0,Wd,stride):
            z = float(depth[v,u])
            if not np.isfinite(z) or z<=0 or z>max_depth:
                continue
            # back-project into camera 3D
            Xc = (u - cx) * z / fx
            Yc = (v - cy) * z / fy
            Zc = z
            Pw = T_wc @ np.array([Xc,Yc,Zc,1.0],dtype=np.float32)
            xw,yw,zw = float(Pw[0]), float(Pw[1]), float(Pw[2])
            if zw < z_min or zw > z_max:
                continue
            gx0,gy0 = world_to_grid(cam_o[0], cam_o[1], origin_x, origin_y, res)
            gx1,gy1 = world_to_grid(xw, yw, origin_x, origin_y, res)
            line = bresenham_line(gx0,gy0,gx1,gy1)
            # free along ray
            for (xg,yg) in line[:-1]:
                if 0<=xg<grid.shape[1] and 0<=yg<grid.shape[0]:
                    grid[yg,xg] = np.clip(grid[yg,xg] + l_free, l_min, l_max)
            # occupied at endpoint
            xg,yg = line[-1]
            if 0<=xg<grid.shape[1] and 0<=yg<grid.shape[0]:
                grid[yg,xg] = np.clip(grid[yg,xg] + l_occ, l_min, l_max)


def export_map(grid_logodds:np.ndarray, out_dir:str, res:float, origin_x:float, origin_y:float,
               web_png:bool=False, web_width:int=1024):
    prob = 1.0/(1.0 + np.exp(-grid_logodds))
    os.makedirs(out_dir, exist_ok=True)
    map_png = os.path.join(out_dir,'map.png')
    cv2.imwrite(map_png, (prob*255).astype(np.uint8))
    # ROS YAML (Nav2-compatible)
    yaml_path = os.path.join(out_dir,'map.yaml')
    with open(yaml_path,'w') as f:
        f.write(f"image: map.png")
        f.write(f"resolution: {res}")
        f.write(f"origin: [{origin_x}, {origin_y}, 0.0]")
        f.write(f"occupied_thresh: 0.65")
        f.write(f"free_thresh: 0.196")
        f.write(f"negate: 0")
    if web_png:
        # scale preserving aspect ratio to web_width
        h,w = prob.shape
        scale = web_width/float(w)
        web_img = cv2.resize((prob*255).astype(np.uint8), (web_width, int(round(h*scale))), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(out_dir,'map_web.png'), web_img)

# ------------------------- Main -------------------------

def main():
    args = parse_args()
    fx,fy,cx,cy = load_intrinsics(args.intrinsics)
    K = (fx,fy,cx,cy)
    poses = load_poses_csv(args.poses)
    res = float(args.grid_res)
    gw = int(np.ceil(args.grid_width / res))
    gh = int(np.ceil(args.grid_height / res))
    grid = np.zeros((gh, gw), dtype=np.float32)

    # Enumerate depth files by frame index or timestamp
    depth_files = [f for f in os.listdir(args.depth_dir) if f.endswith('.npy') and ('depth' in f)]

    def frame_key_from_name(name:str) -> Optional[str]:
        base = os.path.splitext(name)[0]
        # Common cases:
        # 1) frame_000123_depth -> key "000123"
        # 2) 1668016825.126333_depth -> key "1668016825.126333"
        if base.startswith('frame_') and base.endswith('_depth'):
            return base[len('frame_'): -len('_depth')]
        # generic: take prefix before first '_depth'
        i = base.rfind('_depth')
        if i != -1:
            return base[:i]
        return None

    # Build mapping from pose indices to T_wc
    # We assume 'frame_idx' in CSV corresponds either to integer index or timestamp string stripped of non-digits except dot.
    pose_map: Dict[str, np.ndarray] = {}
    for idx, T in poses.items():
        pose_map[str(idx)] = T

    # Sort depth files by their key for reproducibility
    indexed = []
    for fname in depth_files:
        key = frame_key_from_name(fname)
        if key is not None:
            indexed.append((key, fname))
    indexed.sort(key=lambda x: x[0])

    processed = 0
    for key, fname in indexed:
        # Prefer exact key match; fallback to int(key) if possible
        T = None
        if key in pose_map:
            T = pose_map[key]
        else:
            try:
                T = pose_map[str(int(float(key)))]  # handle timestamps like "1668016825.126333"
            except Exception:
                T = None
        if T is None:
            continue
        depth = np.load(os.path.join(args.depth_dir, fname))
        if args.depth_scale != 1.0:
            depth = depth * args.depth_scale
        if depth.ndim != 2:
            if depth.ndim==3 and depth.shape[2]==1:
                depth = depth[:,:,0]
            else:
                raise ValueError(f"Depth file must be 2D array: {fname}")
        integrate_frame(depth, K, T.astype(np.float32), grid, res, args.origin_x, args.origin_y,
                        args.l_free, args.l_occ, args.l_min, args.l_max,
                        args.stride, args.max_depth, args.z_min, args.z_max)
        processed += 1
        if processed % 10 == 0:
            print(f"Processed {processed} frames...")

    os.makedirs(args.out_dir, exist_ok=True)
    export_map(grid, args.out_dir, res, args.origin_x, args.origin_y,
               web_png=args.web_png, web_width=args.web_width)
    print(f"Done. Map saved to {args.out_dir}/map.png and {args.out_dir}/map.yaml")
    if args.web_png:
        print(f"Web PNG: {args.out_dir}/map_web.png")

if __name__ == '__main__':
    main()
