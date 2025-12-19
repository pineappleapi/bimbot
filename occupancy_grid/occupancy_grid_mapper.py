"""
Occupancy Grid Mapper for 2D SLAM visualization
Converts 3D sparse SLAM map to 2D occupancy grid
WITH ZOOM AND PAN CONTROLS
"""

import numpy as np
import cv2


class OccupancyGridMapper:
    def __init__(self, resolution=0.05, size=600, max_height=2.0, min_height=0.1, distance_multiplier=5.0):
        """
        Create a 2D occupancy grid mapper
        
        Args:
            resolution: meters per cell (e.g., 0.05 = 5cm per cell)
            size: grid size in cells (600x600 = 30m x 30m at 0.05m resolution)
            max_height: maximum height to consider points (filter ceiling points)
            min_height: minimum height to consider points (filter floor noise)
            distance_multiplier: Factor to amplify path distances for visualization (default 2.5)
            
        """
        self.resolution = resolution
        self.size = size
        self.max_height = max_height
        self.min_height = min_height
        self.distance_multiplier = distance_multiplier
        
        # Grid: 0.5 = unknown, 0.0 = free, 1.0 = occupied
        self.grid = np.ones((size, size), dtype=np.float32) * 0.5
        
        # Origin at center of grid
        self.origin = np.array([size // 2, size // 2])
        
        # Track camera path
        self.path = []
        
        # Statistics for debugging
        self.total_updates = 0
        self.valid_path_points = 0
        self.out_of_bounds_count = 0
        
        # Zoom and pan controls
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.dragging = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.window_name = "Occupancy Grid"
        self.mouse_callback_set = False

        
        # Pose history for prediction
        self.last_pose = None
        self.prev_pose = None
        self.last_delta = np.eye(4, dtype=np.float32)   # Twc_prev^-1 @ Twc_last
        self.predicted_frames = 0
        self.max_predicted_frames = 150

        
        print(f"OccupancyGridMapper initialized: {size}x{size} cells, {resolution}m resolution")
        print(f"Coverage area: {size*resolution:.1f}m x {size*resolution:.1f}m")
        print(f"Origin at grid center: {self.origin}")
        print("\n=== CONTROLS ===")
        print("Mouse Wheel: Zoom in/out")
        print("Left Click + Drag: Pan")
        print("Right Click: Reset view")
        print("'r' key: Reset zoom and pan")
        print("================\n")
        
    def world_to_grid(self, x, z):
        """
        Convert world coordinates (x, z) to grid coordinates
        
        Args:
            x: world X coordinate (right in camera frame)
            z: world Z coordinate (forward in camera frame)
            
        Returns:
            (grid_x, grid_y): grid cell coordinates
        """
        amplified_x = x * self.distance_multiplier
        amplified_z = z * self.distance_multiplier
        
        grid_x = int(amplified_x / self.resolution) + self.origin[0]
        grid_y = int(-amplified_z / self.resolution) + self.origin[1]
        return grid_x, grid_y
    
    def is_valid_grid_coord(self, grid_x, grid_y):
        """Check if grid coordinates are within bounds"""
        return 0 <= grid_x < self.size and 0 <= grid_y < self.size
    
    def _predict_pose_cv(self):
        """Constant-velocity predictor: Twc_pred = Twc_last @ last_delta"""
        if self.last_pose is None:
            return None
        return self.last_pose @ self.last_delta

    def _update_motion_model(self, pose):
        if pose is None: return
        if self.last_pose is not None:
            inv_last = np.linalg.inv(self.last_pose)
            self.last_delta = inv_last @ pose
        self.prev_pose = self.last_pose
        self.last_pose = pose

        
    def update(self, pose, map_points, slam_state="OK", timestamp=None):
        """
        Update occupancy grid with current camera pose and visible map points.

        Args:
            pose: 4x4 camera pose matrix (Twc), or None when tracking fails
            map_points: list of MapPoint objects from SLAM
            slam_state: "OK" | "RELOCALIZE" | "LOST" | "NOT_INITIALIZED"
            timestamp: optional float (for overlays/logging)
        """

        # ------------------------------------------------------------
        # Pose handling & prediction
        # ------------------------------------------------------------
        is_predicted = False

        if pose is None:
            if slam_state in ("RELOCALIZE", "LOST"):
                # Only predict for a LIMITED time to avoid ghost walls
                if self.predicted_frames < self.max_predicted_frames:
                    pose = self._predict_pose_cv()
                    is_predicted = pose is not None
                    if is_predicted:
                        self.predicted_frames += 1
                        print(f"[OccupancyGrid] Using predicted pose (frame {self.predicted_frames}/{self.max_predicted_frames})")
                    else:
                        print("[OccupancyGrid] No pose & no prediction; skipping frame.")
                        return
                else:
                    print(f"[OccupancyGrid] Prediction limit reached ({self.max_predicted_frames}), skipping to avoid ghost walls.")
                    return
            else:
                print(f"[OccupancyGrid] Pose None in state '{slam_state}', skipping.")
                return
        else:
            self._update_motion_model(pose)
            self.predicted_frames = 0

        # ------------------------------------------------------------
        # Per-frame bookkeeping
        # ------------------------------------------------------------
        self.total_updates += 1

        cam_x = pose[0, 3]
        cam_y = pose[1, 3]  # height
        cam_z = pose[2, 3]

        cam_grid_x, cam_grid_y = self.world_to_grid(cam_x, cam_z)

        if self.total_updates % 20 == 0:
            print(
                f"[OccupancyGrid] Update {self.total_updates}: "
                f"cam_pos=({cam_x:.2f}, {cam_y:.2f}, {cam_z:.2f}) -> "
                f"grid=({cam_grid_x}, {cam_grid_y})"
            )
            print(
                f"[OccupancyGrid] Path len={len(self.path)}, "
                f"valid={self.valid_path_points}, "
                f"OOB={self.out_of_bounds_count}, "
                f"pred_frames={self.predicted_frames}"
            )

        # ------------------------------------------------------------
        # Store camera path & mark camera free
        # ------------------------------------------------------------
        if self.is_valid_grid_coord(cam_grid_x, cam_grid_y):
            if not self.path or self.path[-1] != (cam_grid_x, cam_grid_y):
                self.path.append((cam_grid_x, cam_grid_y))
                self.valid_path_points += 1

            cv2.circle(self.grid, (cam_grid_x, cam_grid_y), 3, 0.0, -1)
        else:
            self.out_of_bounds_count += 1
            if self.out_of_bounds_count <= 5:
                print(
                    f"[OccupancyGrid] WARNING: Camera out of bounds: "
                    f"({cam_grid_x}, {cam_grid_y})"
                )

        # ------------------------------------------------------------
        # Early exit if no points
        # ------------------------------------------------------------
        if not map_points:
            return

        # ------------------------------------------------------------
        # Confidence scaling
        # ------------------------------------------------------------
        state_is_confident = (slam_state == "OK")
        hit_inc  = 0.15 * (1.0 if state_is_confident else 0.5)
        miss_dec = 0.05 * (1.0 if state_is_confident else 0.5)

        # Disable occupied updates after long prediction
        too_long_prediction    = (self.predicted_frames > self.max_predicted_frames)
        allow_occupied_updates = not too_long_prediction

        # ------------------------------------------------------------
        # Collect occupied endpoints
        # ------------------------------------------------------------
        occupied_cells = []

        for p in map_points:
            if p is None or getattr(p, "is_bad", False):
                continue

            pt_x, pt_y, pt_z = p.pt

            if pt_y < self.min_height or pt_y > self.max_height:
                continue

            pt_grid_x, pt_grid_y = self.world_to_grid(pt_x, pt_z)

            if not self.is_valid_grid_coord(pt_grid_x, pt_grid_y):
                continue

            occupied_cells.append((pt_grid_x, pt_grid_y))

        # Avoid reinforcing the same cell multiple times per frame
        occupied_cells = list(set(occupied_cells))

        # ------------------------------------------------------------
        # Apply grid updates (occupied + free space carving)
        # ------------------------------------------------------------
        for pt_grid_x, pt_grid_y in occupied_cells:
            if allow_occupied_updates:
                self.grid[pt_grid_y, pt_grid_x] = min(
                    1.0,
                    self.grid[pt_grid_y, pt_grid_x] + hit_inc
                )

            if self.is_valid_grid_coord(cam_grid_x, cam_grid_y):
                self._mark_ray_as_free_with_strength(
                    cam_grid_x, cam_grid_y,
                    pt_grid_x, pt_grid_y,
                    miss_dec
                )


    
    def _mark_ray_as_free_with_strength(self, x0, y0, x1, y1, miss_dec):
        """
        Mark cells along ray from (x0,y0) to (x1,y1) as free with a configurable decrement.
        Uses OpenCV's line drawing to generate a mask and applies 'miss_dec' instead of a fixed 0.05.
        """
        ray_mask = np.zeros_like(self.grid)
        cv2.line(ray_mask, (x0, y0), (x1, y1), 1.0, thickness=1)
        free_mask = (ray_mask > 0) & (self.grid > 0.2)
        self.grid[free_mask] = np.maximum(0.0, self.grid[free_mask] - miss_dec)


    def _mark_ray_as_free(self, x0, y0, x1, y1):
        """
        Mark cells along ray from (x0,y0) to (x1,y1) as free using Bresenham's algorithm
        """
        # Use OpenCV's line drawing (fast implementation of Bresenham)
        # Create a temporary mask for the ray
        ray_mask = np.zeros_like(self.grid)
        cv2.line(ray_mask, (x0, y0), (x1, y1), 1.0, thickness=1)
        
        # Only update cells that are unknown or have low occupancy
        free_mask = (ray_mask > 0) & (self.grid > 0.2)
        self.grid[free_mask] = np.maximum(0.0, self.grid[free_mask] - 0.05)
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for zoom and pan"""
        
        # Mouse wheel - Zoom
        if event == cv2.EVENT_MOUSEWHEEL:
            # Zoom in/out
            if flags > 0:  # Scroll up - zoom in
                self.zoom_level *= 1.2
            else:  # Scroll down - zoom out
                self.zoom_level /= 1.2
            
            # Limit zoom range
            self.zoom_level = np.clip(self.zoom_level, 0.5, 10.0)
            print(f"[OccupancyGrid] Zoom: {self.zoom_level:.2f}x")
        
        # Left click - Start dragging
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.last_mouse_x = x
            self.last_mouse_y = y  
        # Release left click - Stop dragging
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
        
        # Mouse move while dragging - Pan
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            dx = x - self.last_mouse_x
            dy = y - self.last_mouse_y
            
            # Apply pan (inverted for natural feel)
            self.pan_x += dx / self.zoom_level
            self.pan_y += dy / self.zoom_level
            
            self.last_mouse_x = x
            self.last_mouse_y = y
        
        # Right click - Reset view
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.reset_view()
    
    def reset_view(self):
        """Reset zoom and pan to default"""
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        print("[OccupancyGrid] View reset")
    
    def get_grid_image(self, draw_path=True, draw_current_pos=True):
        """
        Generate visualization image of the occupancy grid
        
        Args:
            draw_path: whether to draw camera path
            draw_current_pos: whether to highlight current position
            
        Returns:
            grid_img: BGR image for display
        """
        # Convert grid to grayscale: 0=free (white), 0.5=unknown (gray), 1=occupied (black)
        grid_normalized = (1.0 - self.grid) * 255
        grid_normalized = np.clip(grid_normalized, 0, 255).astype(np.uint8)
        
        # Apply colormap for better visualization
        grid_img = cv2.applyColorMap(grid_normalized, cv2.COLORMAP_BONE)
        
        # Draw camera path with THICKER, BRIGHTER line
        if draw_path and len(self.path) > 1:
            path_array = np.array(self.path, dtype=np.int32)
            # Adjust line thickness based on zoom
            thickness = max(2, int(4 * self.zoom_level))
            cv2.polylines(grid_img, [path_array], False, (0, 255, 255), thickness)  # Cyan
            cv2.polylines(grid_img, [path_array], False, (0, 200, 0), max(1, thickness // 2))  # Green
        
        # Draw current position with HIGH CONTRAST
        if draw_current_pos and len(self.path) > 0:
            current_pos = self.path[-1]
            radius = max(4, int(8 * self.zoom_level))
            cv2.circle(grid_img, current_pos, radius + 4, (0, 0, 0), 3)      # Black outline
            cv2.circle(grid_img, current_pos, radius + 2, (255, 255, 255), 2) # White ring
            cv2.circle(grid_img, current_pos, radius, (0, 0, 255), -1)       # Red center
        
        # Apply zoom and pan transformation
        if self.zoom_level != 1.0 or self.pan_x != 0 or self.pan_y != 0:
            h, w = grid_img.shape[:2]
            
            # Calculate transformation matrix
            M = np.float32([
                [self.zoom_level, 0, w/2 + self.pan_x - (w/2) * self.zoom_level],
                [0, self.zoom_level, h/2 + self.pan_y - (h/2) * self.zoom_level]
            ])
            
            # Apply transformation
            grid_img = cv2.warpAffine(grid_img, M, (w, h), flags=cv2.INTER_LINEAR)
        
        # Add grid info text with better formatting
        info_text = f"Resolution: {self.resolution*100:.0f}cm | Size: {self.size*self.resolution:.0f}m | Path: {len(self.path)} pts"
        cv2.putText(grid_img, info_text, (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Add zoom and pan info
        zoom_text = f"Zoom: {self.zoom_level:.2f}x | Pan: ({self.pan_x:.0f}, {self.pan_y:.0f})"
        cv2.putText(grid_img, zoom_text, (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
        
        # Add statistics
        stats_text = f"Updates: {self.total_updates} | Valid: {self.valid_path_points} | OOB: {self.out_of_bounds_count}"
        cv2.putText(grid_img, stats_text, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Add controls hint
        controls_text = "Mouse wheel: Zoom | Left drag: Pan | Right click: Reset"
        cv2.putText(grid_img, controls_text, (10, grid_img.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1, cv2.LINE_AA)
        
        # Draw coordinate system indicator (scaled with zoom)
        origin_screen = (int(self.origin[0]), int(self.origin[1]))
        origin_radius = max(3, int(5 * self.zoom_level))
        cv2.circle(grid_img, origin_screen, origin_radius, (255, 0, 255), 2)  # Magenta for origin
        cv2.putText(grid_img, "Origin", (origin_screen[0]+10, origin_screen[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1, cv2.LINE_AA)
        
        return grid_img
    
    def visualize(self, window_name="Occupancy Grid", draw_path=True):
        """
        Display the occupancy grid in a window with zoom and pan controls
        
        Args:
            window_name: name of the display window
            draw_path: whether to draw the camera path
        """
        self.window_name = window_name
        
        # Set up mouse callback only once
        if not self.mouse_callback_set:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(window_name, self._mouse_callback)
            self.mouse_callback_set = True
        
        grid_img = self.get_grid_image(draw_path=draw_path)
        cv2.imshow(window_name, grid_img)
        
        # Handle keyboard shortcuts
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            self.reset_view()
    
    def save(self, filepath):
        """
        Save the occupancy grid as an image
        
        Args:
            filepath: path to save the image (e.g., "map.png")
        """
        # Temporarily reset view for saving full map
        temp_zoom = self.zoom_level
        temp_pan_x = self.pan_x
        temp_pan_y = self.pan_y
        
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        
        grid_img = self.get_grid_image(draw_path=True)
        cv2.imwrite(filepath, grid_img)
        
        # Restore view
        self.zoom_level = temp_zoom
        self.pan_x = temp_pan_x
        self.pan_y = temp_pan_y
        
        print(f"Occupancy grid saved to: {filepath}")
        print(f"Final statistics: Path points={len(self.path)}, Valid={self.valid_path_points}, OOB={self.out_of_bounds_count}")
    
    def reset(self):
        """Reset the occupancy grid and path"""
        self.grid = np.ones((self.size, self.size), dtype=np.float32) * 0.5
        self.path = []
        self.total_updates = 0
        self.valid_path_points = 0
        self.out_of_bounds_count = 0
        self.reset_view()
        print("OccupancyGridMapper reset")

    def get_grid_image_bytes(self, draw_path=True, draw_current_pos=True):
        """Get grid image as JPEG bytes for streaming - SIMPLIFIED VERSION"""
        try:
            # Get the visualization
            grid_img = self.get_grid_image(draw_path=draw_path, draw_current_pos=draw_current_pos)
            
            # Make absolutely sure it's contiguous
            grid_img = np.ascontiguousarray(grid_img)
            
            # Encode with error checking
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            result, encimg = cv2.imencode('.jpg', grid_img, encode_param)
            
            if not result:
                print("[ERROR] imencode returned False")
                return None
                
            data = encimg.tobytes()
            print(f"[OccupancyGrid] Encoded {len(data)} bytes")
            return data
            
        except Exception as e:
            print(f"[ERROR] Encoding failed: {e}")
            import traceback
            traceback.print_exc()
            return None