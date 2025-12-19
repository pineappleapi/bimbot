"""
Shared state for occupancy grid using file-based caching
This avoids import path issues between different directories
"""
from pathlib import Path
import time

GRID_CACHE_FILE = Path("/tmp/bimbot_occupancy_grid.jpg")
GRID_META_FILE = Path("/tmp/bimbot_grid_meta.txt")

def set_grid(data):
    """Save grid data to file cache"""
    try:
        GRID_CACHE_FILE.write_bytes(data)
        GRID_META_FILE.write_text(f"{time.time()}\n{len(data)}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save grid: {e}")
        return False

def get_grid():
    """Read grid data from file cache"""
    try:
        if not GRID_CACHE_FILE.exists():
            return None
        return GRID_CACHE_FILE.read_bytes()
    except Exception as e:
        print(f"[ERROR] Failed to read grid: {e}")
        return None

def get_metadata():
    """Get timestamp and size of last grid update"""
    try:
        if not GRID_META_FILE.exists():
            return None, None
        content = GRID_META_FILE.read_text().strip().split('\n')
        timestamp = float(content[0])
        size = int(content[1])
        return timestamp, size
    except Exception as e:
        return None, None