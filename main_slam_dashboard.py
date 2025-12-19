#!/usr/bin/env -S python3 -O
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

import cv2
import csv
import time
import os
import sys
# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import grid_state
import numpy as np
import json
import gc  # NEW: Added gc import at top

import platform

from pyslam.config import Config  # , dump_config_to_json

from collections import deque
from occupancy_grid import OccupancyGridMapper #NEW: original script added
from performance_logger import RunLogger  #NEW: original script added
from yaw_state_manager import get_yaw_manager #NEW: original script added

from pyslam.semantics.semantic_mapping import SemanticMappingType
from pyslam.semantics.semantic_types import SemanticFeatureType
from pyslam.semantics.semantic_mapping_configs import SemanticMappingConfigs
from pyslam.semantics.semantic_mapping_shared import SemanticMappingShared
from pyslam.semantics.semantic_utils import SemanticDatasetType
from pyslam.semantics.semantic_eval import evaluate_semantic_mapping

from pyslam.slam.slam import Slam, SlamState
from pyslam.viz.slam_plot_drawer import SlamPlotDrawer
from pyslam.slam.camera import PinholeCamera
from pyslam.io.ground_truth import GroundTruthType, groundtruth_factory
from pyslam.io.dataset_factory import dataset_factory
from pyslam.io.dataset_types import DatasetType, SensorType
from pyslam.io.trajectory_writer import TrajectoryWriter

from pyslam.viz.viewer3D import Viewer3D
from pyslam.utilities.utils_sys import getchar, Printer, force_kill_all_and_exit
from pyslam.utilities.utils_img import ImgWriter
from pyslam.utilities.utils_eval import eval_ate
from pyslam.utilities.utils_geom_trajectory import find_poses_associations
from pyslam.utilities.utils_colors import GlColors
from pyslam.utilities.utils_serialization import SerializableEnumEncoder

from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs

from pyslam.loop_closing.loop_detector_configs import LoopDetectorConfigs

from pyslam.depth_estimation.depth_estimator_factory import (
    depth_estimator_factory,
    DepthEstimatorType,
)
from pyslam.utilities.utils_depth import img_from_depth, filter_shadow_points

from pyslam.config_parameters import Parameters

from pyslam.viz.rerun_interface import Rerun

from datetime import datetime
import traceback

import argparse

from matplotlib import pyplot as plt



datetime_string = datetime.now().strftime("%Y%m%d_%H%M%S")

def run_slam(headless: bool = False, config_path: str | None = None, no_output_date: bool = False):
    """
    Programmatic entry point for the SLAM pipeline.

    Args:
        headless: If True, run without GUI (Viewer3D / cv2 windows where applicable).
        config_path: Optional path to a custom config file (same semantics as -c/--config_path).
        no_output_date: If True, do not append date to output directory (same as --no_output_date).

    Behavior mirrors the original CLI, but runs entirely within the current process.
    """
    # --- build "args" equivalent (mirror argparse results) ---
    class Args:
        def __init__(self, headless: bool, config_path: str | None, no_output_date: bool):
            self.headless = headless
            self.config_path = config_path
            self.no_output_date = no_output_date

    args = Args(headless=headless, config_path=config_path, no_output_date=no_output_date)

    # --- original "args-handling" logic lifted from __main__ ---
    # config selection
    if args.config_path:
        config = Config(args.config_path)  # use the custom configuration path file
    else:
        config = Config()

    # datetime string handling (matches --no_output_date)
    global datetime_string  # reuse the module-level datetime_string variable
    if args.no_output_date:
        print("Not appending date to output directory")
        datetime_string = None

    # dataset and basic flags
    dataset = dataset_factory(config)
    is_monocular = dataset.sensor_type == SensorType.MONOCULAR
    num_total_frames = dataset.num_frames

    # trajectory writers
    online_trajectory_writer = None
    final_trajectory_writer = None
    if config.trajectory_saving_settings["save_trajectory"]:
        (
            trajectory_online_file_path,
            trajectory_final_file_path,
            trajectory_saving_base_path,
        ) = config.get_trajectory_saving_paths(datetime_string)

        online_trajectory_writer = TrajectoryWriter(
            format_type=config.trajectory_saving_settings["format_type"],
            filename=trajectory_online_file_path,
        )
        final_trajectory_writer = TrajectoryWriter(
            format_type=config.trajectory_saving_settings["format_type"],
            filename=trajectory_final_file_path,
        )

    metrics_save_dir = trajectory_saving_base_path

    # ground-truth + camera
    groundtruth = groundtruth_factory(config.dataset_settings)
    camera = PinholeCamera(config)

    # feature tracker / loop detector / semantic mapping (unchanged)
    feature_tracker_config = FeatureTrackerConfigs.ORB2_TUNED
    loop_detection_config = LoopDetectorConfigs.DBOW3 
    semantic_mapping_config = (
        SemanticMappingConfigs.get_config_from_slam_dataset(dataset.type)
        if Parameters.kDoSemanticMapping
        else None
    )

    # overrides from settings
    if config.feature_tracker_config_name is not None:
        feature_tracker_config = FeatureTrackerConfigs.get_config_from_name(config.feature_tracker_config_name)

    if config.num_features_to_extract > 0:
        Printer.yellow("Setting feature_tracker_config num_features from settings: ", config.num_features_to_extract)
        feature_tracker_config["num_features"] = config.num_features_to_extract

    if config.loop_detection_config_name is not None:
        loop_detection_config = LoopDetectorConfigs.get_config_from_name(config.loop_detection_config_name)

    if config.semantic_mapping_config_name is not None:
        semantic_mapping_config = SemanticMappingConfigs.get_config_from_name(config.semantic_mapping_config_name)

    Printer.green("feature_tracker_config: ", json.dumps(feature_tracker_config, indent=4, cls=SerializableEnumEncoder))
    Printer.green("loop_detection_config: ", json.dumps(loop_detection_config, indent=4, cls=SerializableEnumEncoder))
    if Parameters.kDoSemanticMapping:
        Printer.green("semantic_mapping_config: ", json.dumps(semantic_mapping_config, indent=4, cls=SerializableEnumEncoder))

    config.feature_tracker_config = feature_tracker_config
    config.loop_detection_config = loop_detection_config
    config.semantic_mapping_config = semantic_mapping_config

    # Select your depth estimator in the front-end (EXPERIMENTAL, WIP)
    depth_estimator = None

    """if is_monocular:
        depth_estimator_type = DepthEstimatorType.DEPTH_ANYTHING_V2
        max_depth = 40
        depth_estimator = depth_estimator_factory(
            depth_estimator_type=depth_estimator_type,
            max_depth=max_depth,
            dataset_env_type=dataset.environmentType(),
            camera=camera,
        )
        Printer.green(f"Depth_estimator_type: {depth_estimator_type.name}, max_depth: {max_depth}")"""

    # create SLAM object
    slam = Slam(
        camera,
        feature_tracker_config,
        loop_detection_config,
        semantic_mapping_config,
        dataset.sensorType(),
        environment_type=dataset.environmentType(),
        config=config,
        headless=args.headless,
    )
    slam.set_viewer_scale(dataset.scale_viewer_3d)
    time.sleep(1)  # to show initial messages

    # load system state if requested
    if config.system_state_load:
        slam.load_system_state(config.system_state_folder_path)
        viewer_scale = (
            slam.viewer_scale() if slam.viewer_scale() > 0 else 0.1
        )  # 0.1 is the default viewer scale
        print(f"viewer_scale: {viewer_scale}")
        slam.set_tracking_state(SlamState.INIT_RELOCALIZE)

    class YawTurnDetector:
        """
        Detects left/right turns from rotation matrices.
        - Smooths yaw over a sliding window.
        - Uses hysteresis to prevent flicker.
        Assumes CV camera coords: +X right, +Y down, +Z forward.
        """
        def __init__(self, enter_deg=5.0, exit_deg=3.0, smooth_window=9):
            assert exit_deg < enter_deg, "exit_deg should be smaller than enter_deg"
            self.enter_deg = enter_deg
            self.exit_deg = exit_deg
            self.window = deque(maxlen=smooth_window)
            self.state = "straight"  # "left" | "right" | "straight"

        @staticmethod
        def yaw_from_R_cv(R: np.ndarray) -> float:
            # yaw = atan2(R[1,0], R[0,0]) under CV camera coords
            return np.degrees(np.arctan2(R[1, 0], R[0, 0]))

        def update(self, R_delta: np.ndarray):
            yaw_deg = self.yaw_from_R_cv(R_delta)
            self.window.append(yaw_deg)
            smoothed = float(np.mean(self.window))
            if self.state == "straight":
                if smoothed >= self.enter_deg:
                    self.state = "left"
                elif smoothed <= -self.enter_deg:
                    self.state = "right"
            elif self.state == "left":
                if smoothed < self.exit_deg:
                    self.state = "straight"
            elif self.state == "right":
                if smoothed > -self.exit_deg:
                    self.state = "straight"
            return self.state, smoothed

    # Create detector, logger, and yaw manager
    turn_detector = YawTurnDetector(enter_deg=5.0, exit_deg=3.0, smooth_window=9)
    yaw_manager = get_yaw_manager()
    performance_logs_name = f"performance_data_{datetime_string or 'run'}.csv"
    run_logger = RunLogger(os.path.join(metrics_save_dir, performance_logs_name))


    # Keep these
    prev_R = None

    # Create CLAHE once (reuse across frames to prevent memory leak)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

    # Initialize occupancy mapper ALWAYS (needed for web streaming)
    occupancy_mapper = OccupancyGridMapper(
        resolution=0.05,      # 5cm per cell
        size=800,             # 800x800 cells = 40m x 40m coverage
        max_height=2.5,       # Ignore ceiling points above 2.5m
        min_height=0.1        # Ignore floor noise below 0.1m
    )

    if args.headless:
        viewer3D = None
        plot_drawer = None
        img_writer = None
        print("Occupancy Grid Mapper initialized (headless mode - web streaming)")
    else:
        viewer3D = None
        plot_drawer = None #SlamPlotDrawer(slam, None) if Parameters.kLocalMappingOnSeparateThread else None
        img_writer = ImgWriter(font_scale=0.7)
        #print("Occupancy Grid Mapper initialized (GUI mode)")
        
        # Create windows only in non-headless mode
        cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Occupancy Grid", cv2.WINDOW_NORMAL)
        

    if groundtruth.type != GroundTruthType.NONE:
        gt_traj3d, gt_poses, gt_timestamps = groundtruth.getFull6dTrajectory()

    do_step = False  # proceed step by step on GUI
    do_reset = False  # reset on GUI
    is_paused = False  # pause/resume on GUI
    is_map_save = False  # save map on GUI
    is_bundle_adjust = False  # bundle adjust on GUI
    is_viewer_closed = False  # viewer GUI was closed

    key = None
    key_cv = None

    num_tracking_lost = 0
    num_frames = 0

    img_id = 0
    while not is_viewer_closed:

        img, img_right, depth = None, None, None

        if do_step:
            Printer.orange("do step: ", do_step)

        if do_reset:
            Printer.yellow("do reset: ", do_reset)
            slam.reset()

        if not is_paused or do_step:

            if dataset.is_ok:
                print("..................................")
                img = dataset.getImageColor(img_id)
                depth = dataset.getDepth(img_id)
                img_right = (
                    dataset.getImageColorRight(img_id)
                    if dataset.sensor_type == SensorType.STEREO
                    else None
                )

            if img is not None:
                timestamp = dataset.getTimestamp()
                next_timestamp = dataset.getNextTimestamp()
                
                frame_duration = (
                    next_timestamp - timestamp
                    if (timestamp is not None and next_timestamp is not None)
                    else -1
                )

                print(f"image: {img_id}, timestamp: {timestamp}, duration: {frame_duration}")

                time_start = time.time()

                # Default values in case pre-tracking fails
                turn, yaw_deg = "straight", 0.0

                # Depth estimation if needed
                if depth is None and depth_estimator:
                    depth_prediction, pts3d_prediction = depth_estimator.infer(img, img_right)
                    if Parameters.kDepthEstimatorRemoveShadowPointsInFrontEnd:
                        depth = filter_shadow_points(depth_prediction)
                    else:
                        depth = depth_prediction
                    print("Depth estimation time: %.3f s" % (time.time() - time_start))
                    if not args.headless:
                        depth_img = img_from_depth(depth_prediction, img_min=0, img_max=50)
                        cv2.imshow("depth prediction", depth_img)

                
                # NEW: --- PREPROCESS: CLAHE for robustness ---
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray_eq = clahe.apply(gray)  # Use pre-created CLAHE
                img_pre = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR) #check

                # ============================================
                # PRE-TRACKING: UPDATE YAW & FEATURES
                # ============================================
                try:
                    from pyslam.slam.frame import FeatureTrackerShared
                except (ModuleNotFoundError, ImportError):
                    FeatureTrackerShared = None

                if FeatureTrackerShared is not None:
                    try:
                        # Get rotation from PREVIOUS frame (before current tracking)
                        prev_R_current = getattr(slam.tracking, "cur_R", None)
                        turn, yaw_deg = "straight", 0.0

                        # Update yaw using previous frame's rotation
                        if prev_R_current is not None:
                            # Calculate yaw change from previous-previous to previous
                            yaw_manager.update_from_rotation(prev_R_current)
                            yaw_deg = yaw_manager.smoothed_yaw_deg
                            
                            # Update turn detector if we have history
                            if prev_R is not None:  # prev_R is from the previous iteration
                                R_delta = prev_R.T @ prev_R_current
                                turn, _ = turn_detector.update(R_delta)
                            
                            print(f"[PRE-TRACK] Yaw: {yaw_deg:+6.2f}° | Turn: {turn:8s}")
                            
                            # Store last good tracking rotation
                            if slam.tracking.state == SlamState.OK:
                                yaw_manager.update_last_good_tracking_R(prev_R_current)
                        
                        # ✅ Set yaw BEFORE tracking
                        FeatureTrackerShared.feature_tracker.setYawDeg(yaw_deg)
                        
                        # Adaptive feature count
                        is_relocalize = (slam.tracking.state == SlamState.RELOCALIZE)
                        
                        if is_relocalize:
                            FeatureTrackerShared.feature_tracker.feature_manager.setMaxFeatures(20000)
                            print("[RELOCALIZE] Boosting to 20k features")
                        elif abs(yaw_deg) >= 8.0:
                            FeatureTrackerShared.feature_tracker.feature_manager.setMaxFeatures(15000)
                            print(f"[TURN] High yaw: 15k features")
                        else:
                            FeatureTrackerShared.feature_tracker.set_normal_num_features()
                        
                        # Update prev_R for next iteration
                        if prev_R_current is not None:
                            prev_R = prev_R_current.copy()
                            
                    except Exception as e:
                        print(f"[warn] pre-track setup failed: {e}")
                        import traceback
                        traceback.print_exc()

                # ============================================
                # MAIN SLAM TRACKING
                # ============================================
                slam.track(img_pre, img_right, depth, img_id, timestamp)

                # ================
                # LOG COMPREHENSIVE PERFORMANCE DATA
                # ================
                run_logger.log(
                    slam=slam,
                    frame_id=img_id,
                    timestamp=timestamp,
                    turn_state=turn,
                    yaw_deg=yaw_deg
                )

                # ============
                # DEBUG: Verify yaw propagation
                # ============
                print(f"[DEBUG] Yaw: {yaw_manager.smoothed_yaw_deg:.1f}°, "
                      f"Features: {len(slam.tracking.f_cur.kps) if slam.tracking.f_cur else 0}")

                
                # ===================
                # UPDATE OCCUPANCY GRID
                # ===================
                if occupancy_mapper and img_id % 2 == 0:
                    try:
                        state_str = {
                            SlamState.OK: "OK",
                            SlamState.RELOCALIZE: "RELOCALIZE",
                            SlamState.LOST: "LOST",
                            SlamState.INIT_RELOCALIZE: "RELOCALIZE"
                        }.get(slam.tracking.state, "OK")

                        current_pose = None
                        cur_R = getattr(slam.tracking, "cur_R", None)
                        cur_t = getattr(slam.tracking, "cur_t", None)

                        if cur_R is not None and cur_t is not None:
                            Twc = np.eye(4, dtype=np.float32)
                            Twc[:3, :3] = cur_R
                            Twc[:3, 3] = cur_t.reshape(3)
                            current_pose = Twc
                        else:
                            if slam.map.num_frames() > 0:
                                last_frame = slam.map.get_frame(-1)
                                current_pose = getattr(last_frame, "Twc", None)

                        map_points = slam.map.get_points()

                        # Update occupancy grid
                        occupancy_mapper.update(
                            current_pose,
                            map_points,
                            slam_state=state_str,
                            timestamp=timestamp
                        )

                        # Export for web interface
                        if headless:
                            try:
                                grid_bytes = occupancy_mapper.get_grid_image_bytes()
                                
                                if grid_bytes is not None:
                                    success = grid_state.set_grid(grid_bytes)
                                    if success and img_id % 20 == 0:
                                        print(f"[OccupancyGrid] Frame {img_id}: Saved {len(grid_bytes)} bytes")
                                else:
                                    print(f"[OccupancyGrid] Frame {img_id}: Failed to encode grid")
                                    
                            except Exception as e:
                                print(f"[ERROR] Failed to export grid: {e}")
                                import traceback
                                traceback.print_exc()
                        
                        occupancy_mapper.visualize()
                    
                    except Exception as e:
                        print(f"[ERROR] Occupancy grid update failed: {e}")
                        import traceback
                        traceback.print_exc()


                # ============================================
                # DISPLAY CAMERA FEED
                # ============================================
                if not args.headless:
                    img_draw = slam.map.draw_feature_trails(img)
                    
                    # Optional: Debug feature distribution (only when needed)
                    # Uncomment the next block to visualize grid and features
                    
                    if img_id % 10 == 0:
                        h, w = img_draw.shape[:2]
                        # Draw grid
                        for i in range(1, 6):
                            cv2.line(img_draw, (0, i*h//6), (w, i*h//6), (0, 255, 0), 1)
                        for j in range(1, 8):
                            cv2.line(img_draw, (j*w//8, 0), (j*w//8, h), (0, 255, 0), 1)
                        # Draw features
                        if hasattr(slam.tracking, 'f_cur') and slam.tracking.f_cur is not None:
                            kps = slam.tracking.f_cur.kps
                            if kps is not None and len(kps) > 0:
                                kps_array = np.array(kps) if not isinstance(kps, np.ndarray) else kps
                                for kp in kps_array:
                                    if hasattr(kp, 'pt'):
                                        x, y = int(kp.pt[0]), int(kp.pt[1])
                                    else:
                                        x, y = int(kp[0]), int(kp[1])
                                    cv2.circle(img_draw, (x, y), 2, (0, 0, 255), -1)
                    
                    img_writer.write(img_draw, f"id: {img_id}", (30, 30))
                    cv2.imshow("Camera", img_draw)

                # Draw plots
                if plot_drawer:
                    plot_drawer.draw(img_id)

                # Save online trajectory
                if (online_trajectory_writer is not None
                    and slam.tracking.cur_R is not None
                    and slam.tracking.cur_t is not None):
                    online_trajectory_writer.write_trajectory(
                        slam.tracking.cur_R, slam.tracking.cur_t, timestamp
                    )

                # Frame timing
                processing_duration = time.time() - time_start
                if frame_duration > processing_duration:
                    time.sleep(frame_duration - processing_duration)

                img_id += 1
                num_frames += 1
            else:
                time.sleep(0.1)
                if args.headless:
                    break

        else:
            time.sleep(0.1)  # Paused

        # ============================================
        # KEYBOARD CONTROLS (non-headless only)
        # ============================================
        if not args.headless:
            # Handle SLAM state changes
            if slam.tracking.state == SlamState.LOST:
                key_cv = cv2.waitKey(500) & 0xFF
            else:
                key_cv = cv2.waitKey(1) & 0xFF
            
            # Get plot drawer key if available
            key = plot_drawer.get_key() if plot_drawer else None
            
            # Process keyboard input
            if key_cv == ord('p') or (key and key == 'p'):
                is_paused = not is_paused
                if is_paused:
                    print("=" * 50)
                    print("PAUSED - Press 'p' to resume")
                    print("=" * 50)
                    
                    # Save occupancy grid when paused
                    if occupancy_mapper:
                        grid_save_path = os.path.join(metrics_save_dir, "occupancy_grid_paused.png")
                        occupancy_mapper.save(grid_save_path)
                        Printer.green(f"Occupancy grid saved to: {grid_save_path}")
            
            elif key_cv == ord('s') or (key and key == 's'):
                if occupancy_mapper:
                    grid_save_path = os.path.join(metrics_save_dir, f"occupancy_grid_snapshot_{img_id}.png")
                    occupancy_mapper.save(grid_save_path)
                    Printer.green(f"Snapshot saved to: {grid_save_path}")
            
            elif key_cv == ord('r') or (key and key == 'r'):
                do_reset = True
            
            elif key_cv == ord('q') or key_cv == 27 or (key and key == 'q'):
                print("Quit requested...")
                break

        # Track lost frames
        if slam.tracking.state == SlamState.LOST:
            num_tracking_lost += 1

    # ============================================
    # CLEANUP & FINAL SAVES
    # ============================================
    print("\n" + "=" * 60)
    print("SHUTTING DOWN...")
    print("=" * 60)

    # Save online trajectory
    if online_trajectory_writer:
        online_trajectory_writer.close_file()

    # Save final occupancy grid
    if occupancy_mapper:
        final_grid_path = os.path.join(metrics_save_dir, "occupancy_grid_final.png")
        occupancy_mapper.save(final_grid_path)
        Printer.green(f"Final occupancy grid saved to: {final_grid_path}")

    # Close run logger
    if run_logger:
        run_logger.close()

    # Quit SLAM
    slam.quit()
    
    if plot_drawer:
        plot_drawer.quit()

    if not args.headless:
        cv2.destroyAllWindows()

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SESSION SUMMARY")
    print("=" * 60)
    print(f"Total frames processed: {num_frames}/{num_total_frames}")
    print(f"Tracking lost: {num_tracking_lost} frames ({num_tracking_lost/max(num_frames,1)*100:.1f}%)")
    print(f"Results saved to: {metrics_save_dir}")
    print("=" * 60)

    if args.headless:
        force_kill_all_and_exit(verbose=False)


# --- preserve the original CLI behavior ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, default=None, help="Optional path for custom configuration file")
    parser.add_argument("--no_output_date", action="store_true", help="Do not append date to output directory")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    args = parser.parse_args()

    run_slam(
        headless=args.headless,
        config_path=args.config_path,
        no_output_date=bool(args.no_output_date),
    )
