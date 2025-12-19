"""
Performance Logger for SLAM
Tracks comprehensive metrics for algorithm evaluation without ground truth
"""

import csv
import numpy as np


class RunLogger:
    def __init__(self, path="logs.csv"):
        self.f = open(path, "w", newline="")
        self.w = csv.writer(self.f)
        
        # Comprehensive header for performance evaluation
        self.w.writerow([
            # Frame identification
            "frame_id",
            "timestamp",
            
            # Feature detection performance
            "num_features_detected",
            "num_features_matched",
            "match_ratio",
            "num_inliers",
            "inlier_ratio",
            
            # Tracking quality
            "mean_reproj_error_px",
            "max_reproj_error_px",
            "median_reproj_error_px",
            "num_outliers",
            
            # Motion estimation
            "translation_norm_m",
            "rotation_angle_deg",
            "yaw_deg",
            "pitch_deg",
            "roll_deg",
            "turn_state",
            
            # SLAM state
            "slam_state",
            "tracking_lost",
            "relocalization_attempted",
            "relocalization_success",
            
            # Map state
            "num_map_points",
            "num_keyframes",
            "num_new_map_points",
            "num_culled_map_points",
            "map_density",
            
            # Keyframe management
            "is_keyframe",
            "keyframe_decision_reason",
            "time_since_last_keyframe_s",
            
            # Performance metrics
            "tracking_time_ms",
            "feature_detection_time_ms",
            "pose_optimization_time_ms",
            "fps",
            
            # Loop closure (if applicable)
            "loop_detected",
            "loop_closure_performed",
            
            # BA statistics
            "local_ba_error",
            "num_optimized_keyframes",
            
            # Memory/efficiency
            "num_active_features",
            "feature_age_avg",
        ])
        
        print(f"Performance RunLogger initialized: {path}")
        
        # Internal state for delta calculations
        self.prev_timestamp = None
        self.prev_num_map_points = 0
        self.last_keyframe_timestamp = 0.0
    
    def log(self, slam, frame_id, timestamp, turn_state="straight", yaw_deg=0.0):
        """
        Automatically extract comprehensive metrics from SLAM object
        
        Args:
            slam: The SLAM object
            frame_id: Current frame ID
            timestamp: Current timestamp
            turn_state: Current turn state (left/right/straight)
            yaw_deg: Current yaw angle in degrees
        """
        
        # ========================================
        # EXTRACT DATA FROM SLAM
        # ========================================
        
        # Feature detection
        num_features_detected = 0
        num_features_matched = 0
        num_inliers = 0
        num_outliers = 0
        num_active_features = 0
        feature_age_avg = 0.0
        
        if hasattr(slam.tracking, 'f_cur') and slam.tracking.f_cur is not None:
            f_cur = slam.tracking.f_cur
            
            # Features detected
            if f_cur.kps is not None:
                num_features_detected = len(f_cur.kps)
            
            # Features matched to map
            if f_cur.points is not None:
                num_features_matched = np.count_nonzero(f_cur.points != None)
                num_inliers = np.count_nonzero(~f_cur.outliers) if hasattr(f_cur, 'outliers') else num_features_matched
                num_outliers = num_features_matched - num_inliers
                
                # Active features (tracked for multiple frames)
                num_active_features = num_features_matched
                
                # Feature age (frames since first observation)
                try:
                    ages = [p.num_observations for p in f_cur.points if p is not None]
                    feature_age_avg = np.mean(ages) if ages else 0.0
                except:
                    feature_age_avg = 0.0
        
        # Match and inlier ratios
        match_ratio = num_features_matched / max(num_features_detected, 1)
        inlier_ratio = num_inliers / max(num_features_matched, 1)
        
        # Reprojection errors
        mean_reproj_error = 0.0
        max_reproj_error = 0.0
        median_reproj_error = 0.0
        
        if hasattr(slam.tracking, 'mean_squared_reproj_err'):
            mean_reproj_error = np.sqrt(slam.tracking.mean_squared_reproj_err)
        elif hasattr(slam.tracking, 'last_optimization_error'):
            mean_reproj_error = slam.tracking.last_optimization_error
        
        # Get individual reprojection errors if available
        try:
            if hasattr(slam.tracking, 'reproj_errors') and slam.tracking.reproj_errors is not None:
                errors = slam.tracking.reproj_errors
                mean_reproj_error = np.mean(errors)
                max_reproj_error = np.max(errors)
                median_reproj_error = np.median(errors)
        except:
            pass
        
        # Motion estimation
        translation_norm = 0.0
        rotation_angle = 0.0
        pitch_deg = 0.0
        roll_deg = 0.0
        
        if hasattr(slam.tracking, 'cur_R') and hasattr(slam.tracking, 'cur_t'):
            if slam.tracking.cur_R is not None and slam.tracking.cur_t is not None:
                # Translation magnitude
                translation_norm = float(np.linalg.norm(slam.tracking.cur_t))
                
                # Total rotation angle
                try:
                    R = slam.tracking.cur_R
                    trace = np.trace(R)
                    rotation_angle = np.degrees(np.arccos(np.clip((trace - 1) / 2, -1, 1)))
                    
                    # Extract Euler angles (pitch, roll)
                    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
                    if sy > 1e-6:
                        pitch_deg = np.degrees(np.arctan2(R[2,1], R[2,2]))
                        roll_deg = np.degrees(np.arctan2(-R[2,0], sy))
                    else:
                        pitch_deg = np.degrees(np.arctan2(-R[1,2], R[1,1]))
                        roll_deg = 0.0
                except:
                    pass
        
        # SLAM state
        slam_state_str = str(slam.tracking.state).split('.')[-1] if hasattr(slam.tracking, 'state') else "UNKNOWN"
        tracking_lost = 1 if slam_state_str == "LOST" else 0
        relocalization_attempted = 1 if slam_state_str == "RELOCALIZE" else 0
        relocalization_success = 1 if (slam_state_str == "OK" and 
                                      hasattr(slam.tracking, 'just_relocated') and 
                                      slam.tracking.just_relocated) else 0
        
        # Map state
        num_map_points = len(slam.map.get_points())
        num_keyframes = slam.map.num_keyframes()
        
        # New/culled points (delta from previous frame)
        num_new_map_points = max(0, num_map_points - self.prev_num_map_points)
        num_culled_map_points = max(0, self.prev_num_map_points - num_map_points)
        self.prev_num_map_points = num_map_points
        
        # Map density (points per keyframe)
        map_density = num_map_points / max(num_keyframes, 1)
        
        # Keyframe management
        is_keyframe = 0
        keyframe_decision_reason = "none"
        if hasattr(slam.tracking, 'need_new_keyframe'):
            is_keyframe = 1 if slam.tracking.need_new_keyframe else 0
            if hasattr(slam.tracking, 'keyframe_decision_reason'):
                keyframe_decision_reason = slam.tracking.keyframe_decision_reason
        
        # Time since last keyframe
        time_since_last_kf = 0.0
        if is_keyframe:
            time_since_last_kf = timestamp - self.last_keyframe_timestamp
            self.last_keyframe_timestamp = timestamp
        else:
            time_since_last_kf = timestamp - self.last_keyframe_timestamp
        
        # Performance timing - FIX: Add None checks
        tracking_time_ms = 0.0
        feature_detection_time_ms = 0.0
        pose_optimization_time_ms = 0.0
        fps = 0.0
        
        if hasattr(slam.tracking, 'time_track') and slam.tracking.time_track is not None:
            tracking_time_ms = slam.tracking.time_track * 1000
        
        if hasattr(slam.tracking, 'time_feat_detection') and slam.tracking.time_feat_detection is not None:
            feature_detection_time_ms = slam.tracking.time_feat_detection * 1000
        
        if hasattr(slam.tracking, 'time_pose_opt') and slam.tracking.time_pose_opt is not None:
            pose_optimization_time_ms = slam.tracking.time_pose_opt * 1000
        
        # FPS calculation
        if self.prev_timestamp is not None and timestamp > self.prev_timestamp:
            fps = 1.0 / (timestamp - self.prev_timestamp)
        self.prev_timestamp = timestamp
        
        # Loop closure
        loop_detected = 0
        loop_closure_performed = 0
        if hasattr(slam, 'loop_closing') and slam.loop_closing is not None:
            if hasattr(slam.loop_closing, 'loop_detected'):
                loop_detected = 1 if slam.loop_closing.loop_detected else 0
            if hasattr(slam.loop_closing, 'loop_closed'):
                loop_closure_performed = 1 if slam.loop_closing.loop_closed else 0
        
        # BA statistics
        local_ba_error = 0.0
        num_optimized_keyframes = 0
        if hasattr(slam, 'local_mapping') and hasattr(slam.local_mapping, 'mean_ba_chi2_error'):
            local_ba_error = slam.local_mapping.mean_ba_chi2_error if slam.local_mapping.mean_ba_chi2_error else 0.0
        
        if hasattr(slam, 'local_mapping') and hasattr(slam.local_mapping, 'last_num_optimized_kfs'):
            num_optimized_keyframes = slam.local_mapping.last_num_optimized_kfs
        
        # ========================================
        # WRITE TO CSV
        # ========================================
        self.w.writerow([
            # Frame identification
            frame_id,
            f"{timestamp:.6f}",
            
            # Feature detection performance
            num_features_detected,
            num_features_matched,
            f"{match_ratio:.4f}",
            num_inliers,
            f"{inlier_ratio:.4f}",
            
            # Tracking quality
            f"{mean_reproj_error:.4f}",
            f"{max_reproj_error:.4f}",
            f"{median_reproj_error:.4f}",
            num_outliers,
            
            # Motion estimation
            f"{translation_norm:.6f}",
            f"{rotation_angle:.4f}",
            f"{yaw_deg:.4f}",
            f"{pitch_deg:.4f}",
            f"{roll_deg:.4f}",
            turn_state,
            
            # SLAM state
            slam_state_str,
            tracking_lost,
            relocalization_attempted,
            relocalization_success,
            
            # Map state
            num_map_points,
            num_keyframes,
            num_new_map_points,
            num_culled_map_points,
            f"{map_density:.2f}",
            
            # Keyframe management
            is_keyframe,
            keyframe_decision_reason,
            f"{time_since_last_kf:.4f}",
            
            # Performance metrics
            f"{tracking_time_ms:.2f}",
            f"{feature_detection_time_ms:.2f}",
            f"{pose_optimization_time_ms:.2f}",
            f"{fps:.2f}",
            
            # Loop closure
            loop_detected,
            loop_closure_performed,
            
            # BA statistics
            f"{local_ba_error:.4f}",
            num_optimized_keyframes,
            
            # Memory/efficiency
            num_active_features,
            f"{feature_age_avg:.2f}",
        ])
    
    def close(self):
        self.f.close()
        print("Performance RunLogger closed")