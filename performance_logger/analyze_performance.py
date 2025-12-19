#!/usr/bin/env python3
"""
Analyze SLAM performance from run logs (no ground truth needed)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

def load_run_data(csv_path):
    """Load run data from CSV"""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} frames from {csv_path}")
    return df

def compute_performance_metrics(df):
    """Compute aggregate performance metrics"""
    
    metrics = {}
    
    # ========================================
    # TRACKING ROBUSTNESS
    # ========================================
    metrics['total_frames'] = len(df)
    metrics['tracking_success_rate'] = (df['slam_state'] == 'OK').sum() / len(df) * 100
    metrics['tracking_lost_frames'] = (df['tracking_lost'] == 1).sum()
    metrics['tracking_lost_rate'] = metrics['tracking_lost_frames'] / len(df) * 100
    metrics['relocalization_attempts'] = (df['relocalization_attempted'] == 1).sum()
    metrics['relocalization_success_rate'] = (
        (df['relocalization_success'] == 1).sum() / max(metrics['relocalization_attempts'], 1) * 100
    )
    
    # ========================================
    # FEATURE TRACKING QUALITY
    # ========================================
    metrics['avg_features_detected'] = df['num_features_detected'].mean()
    metrics['avg_features_matched'] = df['num_features_matched'].mean()
    metrics['avg_match_ratio'] = df['match_ratio'].mean() * 100
    metrics['avg_inliers'] = df['num_inliers'].mean()
    metrics['avg_inlier_ratio'] = df['inlier_ratio'].mean() * 100
    metrics['min_inliers'] = df['num_inliers'].min()
    
    # ========================================
    # TRACKING ACCURACY
    # ========================================
    metrics['avg_reproj_error'] = df['mean_reproj_error_px'].mean()
    metrics['median_reproj_error'] = df['mean_reproj_error_px'].median()
    metrics['max_reproj_error'] = df['max_reproj_error_px'].max()
    metrics['reproj_error_std'] = df['mean_reproj_error_px'].std()
    
    # ========================================
    # MAP QUALITY
    # ========================================
    metrics['final_map_points'] = df['num_map_points'].iloc[-1]
    metrics['final_keyframes'] = df['num_keyframes'].iloc[-1]
    metrics['avg_map_density'] = df['map_density'].mean()
    metrics['total_points_created'] = df['num_new_map_points'].sum()
    metrics['total_points_culled'] = df['num_culled_map_points'].sum()
    metrics['point_survival_rate'] = (
        (metrics['final_map_points'] / max(metrics['total_points_created'], 1)) * 100
    )
    
    # ========================================
    # KEYFRAME MANAGEMENT
    # ========================================
    metrics['num_keyframes_created'] = (df['is_keyframe'] == 1).sum()
    metrics['avg_time_between_keyframes'] = df[df['is_keyframe'] == 1]['time_since_last_keyframe_s'].mean()
    metrics['keyframe_rate'] = metrics['num_keyframes_created'] / len(df) * 100
    
    # ========================================
    # PERFORMANCE
    # ========================================
    metrics['avg_tracking_time_ms'] = df['tracking_time_ms'].mean()
    metrics['avg_fps'] = df['fps'].mean()
    metrics['avg_feature_detection_time_ms'] = df['feature_detection_time_ms'].mean()
    metrics['avg_pose_opt_time_ms'] = df['pose_optimization_time_ms'].mean()
    
    # ========================================
    # MOTION ANALYSIS
    # ========================================
    metrics['total_distance_traveled_m'] = df['translation_norm_m'].sum()
    metrics['avg_velocity_m_s'] = df['translation_norm_m'].mean() / df['timestamp'].diff().mean()
    metrics['max_rotation_deg'] = df['rotation_angle_deg'].max()
    metrics['num_turns'] = len(df[df['turn_state'] != 'straight'])
    metrics['turn_percentage'] = metrics['num_turns'] / len(df) * 100
    
    # ========================================
    # LOOP CLOSURE (if applicable)
    # ========================================
    metrics['loop_closures_detected'] = (df['loop_detected'] == 1).sum()
    metrics['loop_closures_performed'] = (df['loop_closure_performed'] == 1).sum()
    
    # ========================================
    # BUNDLE ADJUSTMENT
    # ========================================
    metrics['avg_local_ba_error'] = df['local_ba_error'].mean()
    metrics['avg_optimized_keyframes'] = df['num_optimized_keyframes'].mean()
    
    return metrics

def print_performance_report(metrics):
    """Print comprehensive performance report"""
    
    print("\n" + "=" * 80)
    print("SLAM PERFORMANCE EVALUATION REPORT (NO GROUND TRUTH)")
    print("=" * 80)
    
    print("\n📊 TRACKING ROBUSTNESS")
    print("-" * 80)
    print(f"  Total Frames Processed:        {metrics['total_frames']:,}")
    print(f"  Tracking Success Rate:         {metrics['tracking_success_rate']:.2f}%")
    print(f"  Tracking Lost (frames):        {metrics['tracking_lost_frames']:,} ({metrics['tracking_lost_rate']:.2f}%)")
    print(f"  Relocalization Attempts:       {metrics['relocalization_attempts']:,}")
    print(f"  Relocalization Success Rate:   {metrics['relocalization_success_rate']:.2f}%")
    
    print("\n🎯 FEATURE TRACKING QUALITY")
    print("-" * 80)
    print(f"  Avg Features Detected:         {metrics['avg_features_detected']:.0f}")
    print(f"  Avg Features Matched:          {metrics['avg_features_matched']:.0f}")
    print(f"  Avg Match Ratio:               {metrics['avg_match_ratio']:.2f}%")
    print(f"  Avg Inliers:                   {metrics['avg_inliers']:.0f}")
    print(f"  Avg Inlier Ratio:              {metrics['avg_inlier_ratio']:.2f}%")
    print(f"  Min Inliers:                   {metrics['min_inliers']:.0f}")
    
    print("\n📐 TRACKING ACCURACY")
    print("-" * 80)
    print(f"  Avg Reprojection Error:        {metrics['avg_reproj_error']:.3f} px")
    print(f"  Median Reprojection Error:     {metrics['median_reproj_error']:.3f} px")
    print(f"  Max Reprojection Error:        {metrics['max_reproj_error']:.3f} px")
    print(f"  Reprojection Error Std Dev:    {metrics['reproj_error_std']:.3f} px")
    
    print("\n🗺️  MAP QUALITY")
    print("-" * 80)
    print(f"  Final Map Points:              {metrics['final_map_points']:,}")
    print(f"  Final Keyframes:               {metrics['final_keyframes']:,}")
    print(f"  Avg Map Density:               {metrics['avg_map_density']:.1f} points/keyframe")
    print(f"  Total Points Created:          {metrics['total_points_created']:,}")
    print(f"  Total Points Culled:           {metrics['total_points_culled']:,}")
    print(f"  Point Survival Rate:           {metrics['point_survival_rate']:.2f}%")
    
    print("\n🔑 KEYFRAME MANAGEMENT")
    print("-" * 80)
    print(f"  Keyframes Created:             {metrics['num_keyframes_created']:,}")
    print(f"  Avg Time Between Keyframes:    {metrics['avg_time_between_keyframes']:.3f} s")
    print(f"  Keyframe Rate:                 {metrics['keyframe_rate']:.2f}%")
    
    print("\n⚡ PERFORMANCE")
    print("-" * 80)
    print(f"  Avg Tracking Time:             {metrics['avg_tracking_time_ms']:.2f} ms")
    print(f"  Avg FPS:                       {metrics['avg_fps']:.2f}")
    print(f"  Avg Feature Detection Time:    {metrics['avg_feature_detection_time_ms']:.2f} ms")
    print(f"  Avg Pose Optimization Time:    {metrics['avg_pose_opt_time_ms']:.2f} ms")
    
    print("\n🚗 MOTION ANALYSIS")
    print("-" * 80)
    print(f"  Total Distance Traveled:       {metrics['total_distance_traveled_m']:.2f} m")
    print(f"  Avg Velocity:                  {metrics['avg_velocity_m_s']:.2f} m/s")
    print(f"  Max Rotation:                  {metrics['max_rotation_deg']:.2f}°")
    print(f"  Frames with Turns:             {metrics['num_turns']:,} ({metrics['turn_percentage']:.2f}%)")
    
    print("\n🔄 LOOP CLOSURE")
    print("-" * 80)
    print(f"  Loop Closures Detected:        {metrics['loop_closures_detected']:,}")
    print(f"  Loop Closures Performed:       {metrics['loop_closures_performed']:,}")
    
    print("\n🎯 BUNDLE ADJUSTMENT")
    print("-" * 80)
    print(f"  Avg Local BA Error:            {metrics['avg_local_ba_error']:.4f}")
    print(f"  Avg Optimized Keyframes:       {metrics['avg_optimized_keyframes']:.1f}")
    
    print("\n" + "=" * 80)

def plot_performance_analysis(df, output_dir):
    """Generate comprehensive performance plots"""
    
    sns.set_style("whitegrid")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # ========================================
    # FIGURE 1: Tracking Quality Over Time
    # ========================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Tracking Quality Over Time', fontsize=16, fontweight='bold')
    
    # Features
    axes[0, 0].plot(df['frame_id'], df['num_features_detected'], label='Detected', alpha=0.7)
    axes[0, 0].plot(df['frame_id'], df['num_features_matched'], label='Matched', alpha=0.7)
    axes[0, 0].plot(df['frame_id'], df['num_inliers'], label='Inliers', alpha=0.7)
    axes[0, 0].set_title('Feature Tracking')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Match and inlier ratios
    axes[0, 1].plot(df['frame_id'], df['match_ratio'] * 100, label='Match Ratio', alpha=0.7)
    axes[0, 1].plot(df['frame_id'], df['inlier_ratio'] * 100, label='Inlier Ratio', alpha=0.7)
    axes[0, 1].set_title('Match & Inlier Ratios')
    axes[0, 1].set_ylabel('Percentage (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Reprojection error
    axes[1, 0].plot(df['frame_id'], df['mean_reproj_error_px'], alpha=0.7, color='red')
    axes[1, 0].axhline(y=2.0, color='orange', linestyle='--', label='Warning (2px)')
    axes[1, 0].axhline(y=5.0, color='red', linestyle='--', label='Critical (5px)')
    axes[1, 0].set_title('Reprojection Error')
    axes[1, 0].set_ylabel('Error (pixels)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # SLAM state
    state_numeric = df['slam_state'].map({'OK': 0, 'LOST': 1, 'RELOCALIZE': 0.5, 'INIT': -0.5})
    axes[1, 1].plot(df['frame_id'], state_numeric, alpha=0.7)
    axes[1, 1].set_title('SLAM State')
    axes[1, 1].set_ylabel('State')
    axes[1, 1].set_yticks([-0.5, 0, 0.5, 1])
    axes[1, 1].set_yticklabels(['INIT', 'OK', 'RELOCALIZE', 'LOST'])
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'tracking_quality.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: tracking_quality.png")
    
    # ========================================
    # FIGURE 2: Map Growth & Quality
    # ========================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Map Growth & Quality', fontsize=16, fontweight='bold')
    
    # Map size
    axes[0, 0].plot(df['frame_id'], df['num_map_points'], label='Map Points')
    ax2 = axes[0, 0].twinx()
    ax2.plot(df['frame_id'], df['num_keyframes'], color='orange', label='Keyframes')
    axes[0, 0].set_title('Map Size')
    axes[0, 0].set_ylabel('Map Points', color='blue')
    ax2.set_ylabel('Keyframes', color='orange')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Map density
    axes[0, 1].plot(df['frame_id'], df['map_density'], alpha=0.7)
    axes[0, 1].set_title('Map Density')
    axes[0, 1].set_ylabel('Points per Keyframe')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Point creation/culling
    axes[1, 0].bar(df['frame_id'], df['num_new_map_points'], label='Created', alpha=0.7, color='green')
    axes[1, 0].bar(df['frame_id'], -df['num_culled_map_points'], label='Culled', alpha=0.7, color='red')
    axes[1, 0].set_title('Point Creation & Culling')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Keyframe timing
    kf_frames = df[df['is_keyframe'] == 1]
    axes[1, 1].scatter(kf_frames['frame_id'], kf_frames['time_since_last_keyframe_s'], alpha=0.7)
    axes[1, 1].set_title('Keyframe Timing')
    axes[1, 1].set_ylabel('Time Since Last KF (s)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'map_quality.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: map_quality.png")

    # ========================================
    # FIGURE 3: Performance & Motion
    # ========================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Performance & Motion Analysis', fontsize=16, fontweight='bold')

    # Processing time breakdown
    axes[0, 0].plot(df['frame_id'], df['tracking_time_ms'], label='Total', alpha=0.7)
    axes[0, 0].plot(df['frame_id'], df['feature_detection_time_ms'], label='Feature Detection', alpha=0.7)
    axes[0, 0].plot(df['frame_id'], df['pose_optimization_time_ms'], label='Pose Optimization', alpha=0.7)
    axes[0, 0].set_title('Processing Time Breakdown')
    axes[0, 0].set_ylabel('Time (ms)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # FPS
    axes[0, 1].plot(df['frame_id'], df['fps'], alpha=0.7)
    axes[0, 1].axhline(y=30, color='green', linestyle='--', label='Real-time (30 FPS)')
    axes[0, 1].set_title('Frames Per Second')
    axes[0, 1].set_ylabel('FPS')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Motion
    axes[1, 0].plot(df['frame_id'], df['translation_norm_m'], alpha=0.7, label='Translation')
    ax2 = axes[1, 0].twinx()
    ax2.plot(df['frame_id'], df['rotation_angle_deg'], color='orange', alpha=0.7, label='Rotation')
    axes[1, 0].set_title('Camera Motion')
    axes[1, 0].set_ylabel('Translation (m)', color='blue')
    ax2.set_ylabel('Rotation (deg)', color='orange')
    axes[1, 0].grid(True, alpha=0.3)

    # Turn analysis
    turn_colors = df['turn_state'].map({'straight': 'green', 'left': 'blue', 'right': 'red'})
    axes[1, 1].scatter(df['frame_id'], df['yaw_deg'], c=turn_colors, alpha=0.5, s=10)
    axes[1, 1].axhline(y=5, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].axhline(y=-5, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('Yaw Angle & Turn Detection')
    axes[1, 1].set_ylabel('Yaw (degrees)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'performance_motion.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: performance_motion.png")

    plt.close('all')