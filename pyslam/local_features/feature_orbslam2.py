"""
* This file is part of PYSLAM
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com>
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
"""

import pyslam.config as config
import os
import cv2
import numpy as np

from pyslam.utilities.utils_sys import Printer
from .feature_base import BaseFeature2D
from orbslam2_features import ORBextractor


kVerbose = True


def distribute_features_grid(
    kps,
    des,
    img_shape,
    grid_size=(6, 8),
    max_per_cell=None,
    yaw_deg=0.0,
):
    """Enhanced grid distribution with AGGRESSIVE turn handling"""
    if len(kps) == 0:
        return kps, des

    # MUCH MORE AGGRESSIVE turn detection
    turning = abs(yaw_deg) >= 5.0
    severe_turn = abs(yaw_deg) >= 10.0  # NEW: detect severe turns
    
    # DYNAMIC BOOST based on turn severity
    if severe_turn:
        edge_boost = 2.5  # ← MASSIVE boost for severe turns
    elif turning:
        edge_boost = 1.8  # ← Increased from 1.3
    else:
        edge_boost = 1.0

    h, w = img_shape[:2]
    grid_rows, grid_cols = grid_size
    cell_h, cell_w = h / grid_rows, w / grid_cols

    if max_per_cell is None:
        total_cells = grid_rows * grid_cols
        max_per_cell = max(10, int(len(kps) / total_cells * 1.2))

    grid = [[[] for _ in range(grid_cols)] for _ in range(grid_rows)]

    for i, kp in enumerate(kps):
        x, y = kp.pt
        row = min(int(y / cell_h), grid_rows - 1)
        col = min(int(x / cell_w), grid_cols - 1)
        grid[row][col].append((kp, des[i] if des is not None else None))

    filtered_kps = []
    filtered_des = []

    # INCREASED overflow allowance
    overflow = int(max_per_cell * 0.5)  # ← Increased from 0.3
    limit = int((max_per_cell + overflow) * edge_boost)

    for row in range(grid_rows):
        for col in range(grid_cols):
            cell = grid[row][col]
            if not cell:
                continue

            cell_sorted = sorted(cell, key=lambda x: x[0].response, reverse=True)

            for kp, desc in cell_sorted[:limit]:
                filtered_kps.append(kp)
                if desc is not None:
                    filtered_des.append(desc)

    # RELAXED safety fallback - only trigger if catastrophic loss
    if len(filtered_kps) < 0.4 * len(kps):  # ← Reduced from 0.6
        Printer.red(f"CRITICAL: Grid filter removed {len(kps) - len(filtered_kps)} features, reverting!")
        return kps, des

    if des is not None:
        filtered_des = np.array(filtered_des, dtype=des.dtype)

    return filtered_kps, filtered_des


# Interface for pySLAM
class Orbslam2Feature2D(BaseFeature2D):
    def __init__(self, num_features=2000, scale_factor=1.2, num_levels=8, 
                 use_grid_filter=True, grid_size=(8, 12)):
        """
        Initialize ORB-SLAM2 feature extractor with optional grid filtering.
        
        Args:
            num_features: Target number of features (after filtering if enabled)
            scale_factor: Scale factor between pyramid levels
            num_levels: Number of pyramid levels
            use_grid_filter: Enable grid-based feature distribution
            grid_size: (rows, cols) for grid division
        """
        print("Using Orbslam2Feature2D")
        
        # Request MORE features from C++ extractor since we'll filter them spatially
        # This ensures we have enough features to fill all grid cells
        extractor_features = int(num_features * 1.8) if use_grid_filter else num_features
        self.orb_extractor = ORBextractor(extractor_features, scale_factor, num_levels)
        
        # Grid filtering settings
        self.use_grid_filter = use_grid_filter
        self.grid_size = grid_size
        self.target_num_features = num_features
        self.initial_num_features = num_features
        self.last_yaw_deg = 0.0  # SAFE DEFAULT
        
        if use_grid_filter:
            Printer.green(f"ORB2 Grid Filter: ENABLED with {grid_size[0]}x{grid_size[1]} grid")
        else:
            Printer.yellow(f"ORB2 Grid Filter: DISABLED")

    def setYawDeg(self, yaw_deg):
        self.last_yaw_deg = float(yaw_deg)
        
    # extract keypoints
    def detect(self, img, mask=None):
        kps_tuples = self.orb_extractor.detect(img)
        kps = [cv2.KeyPoint(*kp) for kp in kps_tuples]
        
        # Apply grid filtering
        if self.use_grid_filter and len(kps) > 0:
            original_count = len(kps)
            kps, _ = distribute_features_grid(kps, None, img.shape, 
                                             self.grid_size, None)
            if kVerbose:
                Printer.cyan(f"ORB2 Grid: {original_count} → {len(kps)} features")
        
        return kps

    def compute(self, img, kps, mask=None):
        Printer.orange(
            "WARNING: you are supposed to call detectAndCompute() for ORB2 instead of compute()"
        )
        Printer.orange("WARNING: ORB2 is recomputing both kps and des on input frame", img.shape)
        return self.detectAndCompute(img)

    def setMaxFeatures(self, num_features):
        """Update the number of features to extract"""
        self.target_num_features = num_features
        extractor_features = int(num_features * 1.8) if self.use_grid_filter else num_features
        self.orb_extractor.SetNumFeatures(extractor_features)
        if kVerbose:
            Printer.blue(f"ORB2: Updated to {num_features} target features")

    # compute both keypoints and descriptors
    def detectAndCompute(self, img, mask=None):
        """Enhanced with AGGRESSIVE turn handling"""
        # Detect and compute from C++ extractor
        kps_tuples, des = self.orb_extractor.detectAndCompute(img)
        kps = [cv2.KeyPoint(*kp) for kp in kps_tuples]
        
        if self.use_grid_filter and len(kps) > 0:
            original_count = len(kps)
            
            # NEW: MUCH MORE AGGRESSIVE relaxation conditions
            severe_turn = abs(self.last_yaw_deg) >= 10.0
            weak_tracking = original_count < 2500  # ← Reduced from 3000
            
            if severe_turn or weak_tracking:
                # ULTRA-RELAXED filtering
                relaxed_grid = (3, 4)  # ← COARSER grid (was 4, 6)
                Printer.orange(f"ORB2 AGGRESSIVE MODE: yaw={self.last_yaw_deg:.1f}°, count={original_count}")
                
                kps, des = distribute_features_grid(
                    kps, des, img.shape,
                    grid_size=relaxed_grid,  # Much coarser grid
                    max_per_cell=None,
                    yaw_deg=self.last_yaw_deg
                )
            else:
                # Normal filtering
                kps, des = distribute_features_grid(
                    kps, des, img.shape, 
                    self.grid_size,
                    None,
                    self.last_yaw_deg
                )
            
            if kVerbose:
                color = "red" if severe_turn else "yellow" if weak_tracking else "cyan"
                getattr(Printer, color)(
                    f"ORB2 Grid: {original_count} → {len(kps)} features "
                    f"(yaw={self.last_yaw_deg:.1f}°, mode={'AGGRESSIVE' if (severe_turn or weak_tracking) else 'NORMAL'})"
                )
        
        return kps, des