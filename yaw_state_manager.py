#!/usr/bin/env -S python3 -O
"""
Centralized yaw state management for turn-aware SLAM
"""
import numpy as np
from collections import deque


class YawStateManager:
    """Singleton manager for yaw state across SLAM components"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.current_yaw_deg = 0.0
        self.smoothed_yaw_deg = 0.0
        self.prev_R = None
        self.yaw_history = deque(maxlen=9)
        self.last_good_tracking_R = None  # For relocalization
        self._initialized = True
    
    def update_from_rotation(self, R_current):
        """Update yaw from rotation matrix"""
        if R_current is None:
            return self.current_yaw_deg
        
        yaw_deg = 0.0
        if self.prev_R is not None:
            try:
                R_delta = self.prev_R.T @ R_current
                yaw_deg = float(np.degrees(np.arctan2(R_delta[1, 0], R_delta[0, 0])))
            except Exception:
                yaw_deg = 0.0
        
        self.prev_R = R_current.copy()
        self.current_yaw_deg = yaw_deg
        
        # Update smoothed version
        self.yaw_history.append(yaw_deg)
        if len(self.yaw_history) > 0:
            self.smoothed_yaw_deg = float(np.mean(self.yaw_history))
        
        return self.current_yaw_deg
    
    def update_last_good_tracking_R(self, R):
        """Store rotation from last successful tracking frame"""
        if R is not None:
            self.last_good_tracking_R = R.copy()
    
    def get_yaw_for_relocalization(self):
        """Get yaw estimate for relocalization (from last good tracking)"""
        if self.last_good_tracking_R is None:
            return 0.0
        
        # Use smoothed yaw as best estimate
        return self.smoothed_yaw_deg
    
    def reset(self):
        """Reset all state"""
        self.current_yaw_deg = 0.0
        self.smoothed_yaw_deg = 0.0
        self.prev_R = None
        self.yaw_history.clear()
        self.last_good_tracking_R = None


# Global singleton instance
_yaw_manager = YawStateManager()

def get_yaw_manager():
    return _yaw_manager